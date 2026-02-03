# 라이브러리 없이 Structured Pruning + Weight Quantization 구현 전략

> **과제 주제:** PyTorch 내장 라이브러리(`torch.nn.utils.prune`, `torch.quantization`) 없이 직접 구현

---

## 1. Structured Pruning vs Unstructured Pruning

### 1.1 차이점

| | Unstructured Pruning | Structured Pruning |
|---|---|---|
| 제거 단위 | 개별 가중치 (weight) | 채널/필터 전체 |
| 결과 | Sparse tensor (0이 많음) | 실제 텐서 크기 감소 |
| 속도 향상 | 희소 행렬 연산 필요 | **즉시 속도 향상** |
| 구현 복잡도 | 쉬움 | 중간 |
| 하드웨어 호환 | 특수 하드웨어 필요 | **범용 하드웨어 OK** |

### 1.2 Structured Pruning 선택 이유

```
Unstructured: [1, 0, 3, 0, 0, 2, 0, 4] → 여전히 8개 원소 저장
Structured:   [1, 3, 2, 4] → 실제로 4개만 저장 (메모리 50% 감소)
```

**장점:**
- 실제 모델 크기 감소
- 추론 속도 향상 (일반 GPU/CPU에서)
- 추가 라이브러리 없이 동작

---

## 2. Structured Pruning 직접 구현

### 2.1 핵심 아이디어

Conv layer에서 **출력 채널(filter)**을 제거:
```
Conv2d(in=32, out=64, k=3) → 필터 64개 중 16개 제거 → Conv2d(in=32, out=48, k=3)
```

**주의:** 다음 레이어의 입력 채널도 함께 조정해야 함!

```
Layer N:   Conv2d(32, 64) → Conv2d(32, 48)  ← 출력 채널 16개 제거
Layer N+1: Conv2d(64, 128) → Conv2d(48, 128)  ← 입력 채널도 맞춰야 함!
```

### 2.2 채널 중요도 계산

```python
def get_channel_importance(conv_layer):
    """
    L1 Norm 기반 채널 중요도 계산

    Args:
        conv_layer: nn.Conv2d layer
    Returns:
        importance: shape (out_channels,)
    """
    # weight shape: (out_channels, in_channels, H, W)
    weight = conv_layer.weight.data

    # 각 출력 채널(필터)의 L1 norm 계산
    # dim=(1,2,3): in_channels, H, W 방향으로 합산
    importance = torch.sum(torch.abs(weight), dim=(1, 2, 3))

    return importance


def get_channels_to_prune(importance, prune_ratio):
    """
    제거할 채널 인덱스 반환

    Args:
        importance: 채널별 중요도
        prune_ratio: 제거 비율 (0.0 ~ 1.0)
    Returns:
        prune_indices: 제거할 채널 인덱스
        keep_indices: 유지할 채널 인덱스
    """
    num_channels = len(importance)
    num_prune = int(num_channels * prune_ratio)

    # 중요도 낮은 순으로 정렬
    sorted_indices = torch.argsort(importance)

    prune_indices = sorted_indices[:num_prune]
    keep_indices = sorted_indices[num_prune:]

    # 인덱스 정렬 (원래 순서 유지)
    keep_indices = torch.sort(keep_indices)[0]

    return prune_indices, keep_indices
```

### 2.3 레이어 Pruning 적용

```python
def prune_conv_layer(conv, keep_indices):
    """
    Conv layer의 출력 채널 pruning

    Args:
        conv: nn.Conv2d
        keep_indices: 유지할 채널 인덱스
    Returns:
        new_conv: pruned Conv2d
    """
    in_channels = conv.in_channels
    out_channels = len(keep_indices)

    new_conv = nn.Conv2d(
        in_channels, out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups if conv.groups == 1 else out_channels,  # Depthwise 처리
        bias=conv.bias is not None
    )

    # 가중치 복사 (유지할 채널만)
    new_conv.weight.data = conv.weight.data[keep_indices].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data[keep_indices].clone()

    return new_conv


def prune_bn_layer(bn, keep_indices):
    """
    BatchNorm layer pruning (Conv와 채널 수 맞추기)
    """
    num_features = len(keep_indices)

    new_bn = nn.BatchNorm2d(num_features)
    new_bn.weight.data = bn.weight.data[keep_indices].clone()
    new_bn.bias.data = bn.bias.data[keep_indices].clone()
    new_bn.running_mean = bn.running_mean[keep_indices].clone()
    new_bn.running_var = bn.running_var[keep_indices].clone()

    return new_bn


def prune_next_conv_input(conv, keep_indices):
    """
    다음 Conv layer의 입력 채널 조정
    (이전 레이어 출력 채널이 줄었으므로)
    """
    in_channels = len(keep_indices)
    out_channels = conv.out_channels

    new_conv = nn.Conv2d(
        in_channels, out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        groups=conv.groups,
        bias=conv.bias is not None
    )

    # 입력 채널 방향으로 pruning
    # weight shape: (out, in, H, W)
    new_conv.weight.data = conv.weight.data[:, keep_indices].clone()
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()

    return new_conv
```

### 2.4 MobileNetV2 Block Pruning

MobileNetV2의 Inverted Residual Block 구조:
```
conv1 (expand) → bn1 → relu
conv2 (depthwise) → bn2 → relu
conv3 (project) → bn3 (no relu)
+ shortcut
```

**Pruning 전략:**
1. `conv1` (Pointwise expand) 출력 채널 pruning
2. `conv2` (Depthwise)도 같이 조정 (groups 파라미터!)
3. `conv3` 입력 채널 조정

```python
def prune_mobilenetv2_block(block, prune_ratio):
    """
    MobileNetV2 Block pruning

    Block 구조:
    - conv1: 1x1 expand (in → planes)
    - conv2: 3x3 depthwise (planes → planes, groups=planes)
    - conv3: 1x1 project (planes → out)
    """
    # 1. Expansion layer (conv1) 중요도 계산
    importance = get_channel_importance(block.conv1)
    _, keep_indices = get_channels_to_prune(importance, prune_ratio)

    # 2. conv1 출력 채널 pruning
    block.conv1 = prune_conv_layer(block.conv1, keep_indices)
    block.bn1 = prune_bn_layer(block.bn1, keep_indices)

    # 3. conv2 (Depthwise) - 입력/출력 모두 조정
    # Depthwise는 groups=in_channels 이므로 특별 처리
    new_planes = len(keep_indices)
    block.conv2 = nn.Conv2d(
        new_planes, new_planes,
        kernel_size=3, stride=block.stride, padding=1,
        groups=new_planes, bias=False
    )
    block.conv2.weight.data = block.conv2.weight.data[keep_indices].clone()
    block.bn2 = prune_bn_layer(block.bn2, keep_indices)

    # 4. conv3 입력 채널 조정
    block.conv3 = prune_next_conv_input(block.conv3, keep_indices)

    return block
```

### 2.5 전체 모델 Pruning

```python
def prune_mobilenetv2(model, prune_ratio=0.5):
    """
    전체 MobileNetV2 모델 Structured Pruning
    """
    pruned_model = copy.deepcopy(model)

    for name, module in pruned_model.named_modules():
        if isinstance(module, Block):  # MobileNetV2의 Block
            prune_mobilenetv2_block(module, prune_ratio)

    return pruned_model
```

---

## 3. Weight Quantization 직접 구현

### 3.1 양자화 기본 공식

**FP32 → INT8 변환:**
```
q = round((x - zero_point) / scale)
q = clamp(q, 0, 255)  # uint8의 경우
```

**INT8 → FP32 복원 (Dequantize):**
```
x' = scale * q + zero_point
```

### 3.2 Scale과 Zero Point 계산

```python
def compute_quantization_params(tensor, num_bits=8):
    """
    Asymmetric Quantization 파라미터 계산

    Args:
        tensor: 양자화할 FP32 텐서
        num_bits: 비트 수 (기본 8)
    Returns:
        scale: 스케일 값
        zero_point: 제로 포인트
    """
    qmin = 0
    qmax = 2 ** num_bits - 1  # 255 for 8-bit

    # 텐서의 최소/최대값
    min_val = tensor.min().item()
    max_val = tensor.max().item()

    # Scale 계산
    scale = (max_val - min_val) / (qmax - qmin)

    # Scale이 0이면 (모든 값이 같으면) 작은 값으로 대체
    if scale == 0:
        scale = 1e-8

    # Zero point 계산
    zero_point = qmin - min_val / scale
    zero_point = int(round(zero_point))
    zero_point = max(qmin, min(qmax, zero_point))

    return scale, zero_point


def compute_symmetric_quantization_params(tensor, num_bits=8):
    """
    Symmetric Quantization 파라미터 계산
    (zero_point = 0, signed int 사용)

    Weight quantization에 주로 사용
    """
    qmax = 2 ** (num_bits - 1) - 1  # 127 for 8-bit

    # 절대값 최대
    max_val = tensor.abs().max().item()

    # Scale 계산
    scale = max_val / qmax if max_val != 0 else 1e-8

    return scale, 0  # zero_point는 항상 0
```

### 3.3 가중치 양자화

```python
def quantize_tensor(tensor, scale, zero_point, num_bits=8):
    """
    텐서를 INT로 양자화
    """
    qmin = 0 if zero_point != 0 else -(2 ** (num_bits - 1))
    qmax = 2 ** num_bits - 1 if zero_point != 0 else 2 ** (num_bits - 1) - 1

    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)

    return q.to(torch.int8)


def dequantize_tensor(q_tensor, scale, zero_point):
    """
    INT 텐서를 FP32로 복원
    """
    return scale * (q_tensor.float() - zero_point)
```

### 3.4 Conv Layer 양자화

```python
class QuantizedConv2d(nn.Module):
    """
    양자화된 Conv2d 레이어

    가중치는 INT8로 저장, 추론 시 dequantize 후 연산
    (실제 INT8 연산은 하드웨어 지원 필요)
    """
    def __init__(self, conv_layer):
        super().__init__()

        # 원본 설정 저장
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.groups = conv_layer.groups

        # 가중치 양자화
        weight = conv_layer.weight.data
        self.weight_scale, self.weight_zp = compute_symmetric_quantization_params(weight)
        self.weight_quantized = quantize_tensor(weight, self.weight_scale, self.weight_zp)

        # Bias는 보통 FP32 유지 (또는 INT32)
        self.bias = conv_layer.bias

    def forward(self, x):
        # Dequantize weight
        weight_fp32 = dequantize_tensor(
            self.weight_quantized,
            self.weight_scale,
            self.weight_zp
        )

        # 일반 Conv 연산 수행
        return F.conv2d(
            x, weight_fp32, self.bias,
            self.stride, self.padding, 1, self.groups
        )

    def get_quantized_weight(self):
        """저장용 INT8 가중치 반환"""
        return self.weight_quantized, self.weight_scale, self.weight_zp
```

### 3.5 Activation Quantization (선택적)

가중치뿐 아니라 활성화(Activation)도 양자화하면 더 효율적:

```python
class QuantizedActivation(nn.Module):
    """
    활성화 양자화를 위한 래퍼

    Calibration 단계에서 min/max 수집 후 양자화 적용
    """
    def __init__(self):
        super().__init__()
        self.calibrating = True
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.scale = None
        self.zero_point = None

    def calibrate(self, x):
        """Calibration: 실제 데이터로 min/max 수집"""
        self.min_val = min(self.min_val, x.min().item())
        self.max_val = max(self.max_val, x.max().item())

    def compute_params(self):
        """Calibration 완료 후 파라미터 계산"""
        self.scale = (self.max_val - self.min_val) / 255
        self.zero_point = int(round(-self.min_val / self.scale))
        self.calibrating = False

    def forward(self, x):
        if self.calibrating:
            self.calibrate(x)
            return x
        else:
            # Quantize → Dequantize (시뮬레이션)
            q = torch.round(x / self.scale + self.zero_point)
            q = torch.clamp(q, 0, 255)
            return self.scale * (q - self.zero_point)
```

### 3.6 전체 모델 양자화

```python
def quantize_model(model):
    """
    전체 모델의 Conv 레이어 양자화
    """
    quantized_model = copy.deepcopy(model)

    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 부모 모듈 찾아서 교체
            parent = get_parent_module(quantized_model, name)
            attr_name = name.split('.')[-1]
            setattr(parent, attr_name, QuantizedConv2d(module))

    return quantized_model


def get_parent_module(model, target_name):
    """모듈의 부모 모듈 반환"""
    names = target_name.split('.')
    module = model
    for name in names[:-1]:
        module = getattr(module, name)
    return module
```

---

## 4. 통합 파이프라인

### 4.1 전체 흐름

```python
def compress_mobilenetv2(model, trainloader, testloader,
                         prune_ratio=0.5, num_epochs_finetune=5):
    """
    MobileNetV2 압축 파이프라인

    1. Structured Pruning
    2. Fine-tuning
    3. Weight Quantization
    """

    # ===== Phase 1: Structured Pruning =====
    print("Phase 1: Structured Pruning...")
    pruned_model = prune_mobilenetv2(model, prune_ratio)

    # Pruning 후 정확도 확인
    acc_after_prune = evaluate(pruned_model, testloader)
    print(f"  Accuracy after pruning: {acc_after_prune:.2f}%")

    # ===== Phase 2: Fine-tuning =====
    print("\nPhase 2: Fine-tuning...")
    optimizer = optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs_finetune):
        train_one_epoch(pruned_model, trainloader, optimizer)
        acc = evaluate(pruned_model, testloader)
        print(f"  Epoch {epoch+1}: {acc:.2f}%")

    # ===== Phase 3: Weight Quantization =====
    print("\nPhase 3: Weight Quantization...")
    quantized_model = quantize_model(pruned_model)

    # 양자화 후 정확도 확인
    acc_final = evaluate(quantized_model, testloader)
    print(f"  Final accuracy: {acc_final:.2f}%")

    # ===== 결과 요약 =====
    original_size = get_model_size(model)
    final_size = get_model_size(quantized_model)

    print("\n" + "="*50)
    print("Compression Results:")
    print(f"  Original size: {original_size:.2f} MB")
    print(f"  Final size: {final_size:.2f} MB")
    print(f"  Compression ratio: {original_size/final_size:.2f}x")
    print(f"  Accuracy: {acc_final:.2f}%")

    return quantized_model
```

### 4.2 모델 크기 측정

```python
def get_model_size(model, count_zeros=False):
    """
    모델 크기 계산 (MB)

    Args:
        model: PyTorch 모델
        count_zeros: True면 0도 포함, False면 non-zero만
    """
    total_bytes = 0

    for param in model.parameters():
        if count_zeros:
            num_elements = param.numel()
        else:
            num_elements = (param != 0).sum().item()

        # 데이터 타입별 바이트 수
        if param.dtype == torch.float32:
            bytes_per_element = 4
        elif param.dtype == torch.int8:
            bytes_per_element = 1
        else:
            bytes_per_element = param.element_size()

        total_bytes += num_elements * bytes_per_element

    return total_bytes / (1024 * 1024)  # MB
```

---

## 5. 예상 결과 및 발표 포인트

### 5.1 예상 압축 결과

| 단계 | 파라미터 수 | 크기 (FP32) | 크기 (INT8) | 정확도 |
|------|-----------|-------------|-------------|--------|
| Baseline | 2.3M | 9.0 MB | - | 94% |
| Pruned 50% | 1.15M | 4.5 MB | - | 91% |
| Fine-tuned | 1.15M | 4.5 MB | - | 93% |
| Quantized | 1.15M | - | **1.15 MB** | 92.5% |

**총 압축률:** 9.0 MB → 1.15 MB = **~8배 압축**

### 5.2 발표 핵심 포인트

1. **왜 MobileNetV2인가?**
   - Depthwise Conv: 채널 독립성 → Structured Pruning 용이
   - Linear Bottleneck: 대칭 양자화 적용 가능
   - Expansion layer: Pruning 여유 공간

2. **왜 Structured Pruning인가?**
   - 실제 모델 크기 감소 (Unstructured는 sparse만)
   - 추가 하드웨어/라이브러리 없이 속도 향상
   - 직접 구현 가능

3. **왜 라이브러리 없이?**
   - 원리 이해
   - 커스터마이징 가능
   - 교육적 목적

4. **구현 핵심 난이도**
   - Conv → BN → 다음 Conv의 채널 연결 유지
   - Depthwise Conv의 groups 파라미터 처리
   - Shortcut connection 처리

---

## 6. 구현 체크리스트

- [ ] 채널 중요도 계산 (L1 Norm)
- [ ] Conv layer pruning (출력 채널 제거)
- [ ] BN layer pruning (채널 맞추기)
- [ ] 다음 layer 입력 채널 조정
- [ ] Depthwise Conv 특수 처리
- [ ] Shortcut connection 처리
- [ ] Fine-tuning 루프
- [ ] Weight quantization (Scale/ZP 계산)
- [ ] Quantized Conv layer 구현
- [ ] 모델 크기 측정 함수
- [ ] 전체 파이프라인 통합
