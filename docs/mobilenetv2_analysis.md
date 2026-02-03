# MobileNetV2: Pruning & Quantization 적합성 분석

## 1. MobileNetV2 핵심 구조

### 1.1 Inverted Residual Block (역 병목 구조)

```
Input (narrow) → Expand (wide) → Depthwise Conv → Project (narrow) → Output
     ↓                                                                  ↑
     └──────────────────── Skip Connection ─────────────────────────────┘
```

**일반 Residual (ResNet):**
```
Wide → Narrow → Wide (병목에서 압축)
```

**Inverted Residual (MobileNetV2):**
```
Narrow → Wide → Narrow (확장 후 압축)
```

이 구조가 pruning/quantization에 유리한 이유:
- **Narrow 입출력**: 적은 채널에서 skip connection → 양자화 오차 전파 최소화
- **Wide expansion**: 충분한 표현력 확보 후 압축 → pruning 여유 공간 제공

### 1.2 Depthwise Separable Convolution

```python
# 일반 Conv: 3x3xCxC' 파라미터
nn.Conv2d(C, C', kernel_size=3)  # 파라미터: 9 * C * C'

# Depthwise Separable: 3x3xC + 1x1xCxC' 파라미터
nn.Conv2d(C, C, kernel_size=3, groups=C)  # Depthwise: 9 * C
nn.Conv2d(C, C', kernel_size=1)            # Pointwise: C * C'
# 총 파라미터: 9*C + C*C' (일반 대비 ~8-9배 감소)
```

**Pruning 관점:**
- 채널별 독립적 연산 → 채널 단위 pruning 시 다른 채널에 영향 없음
- 구조적 pruning(Structured Pruning) 적용 용이

**Quantization 관점:**
- 작은 커널(3x3) → 가중치 분포가 단순
- Pointwise(1x1) conv → 행렬곱으로 변환 가능, 하드웨어 가속 용이

### 1.3 Linear Bottleneck

```python
# MobileNetV2 Block의 마지막
self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1)  # Projection
self.bn3 = nn.BatchNorm2d(out_planes)
# ReLU 없음! (Linear)
```

**왜 ReLU를 제거했나?**
- 저차원(narrow) 공간에서 ReLU → 정보 손실 발생
- ReLU는 음수를 0으로 만들어 manifold를 손상시킴
- Linear layer는 정보를 보존

**Quantization 관점:**
- ReLU 없음 → 출력 범위가 음수 포함 (대칭적)
- 대칭 양자화(Symmetric Quantization) 적용 용이

### 1.4 ReLU6 사용

```python
F.relu6(x)  # min(max(0, x), 6)
```

**Quantization에 극도로 유리:**
- 출력이 [0, 6]으로 제한됨
- 양자화 시 클리핑 범위가 명확
- INT8 양자화 시 최적의 해상도 확보

---

## 2. 다른 모델들과의 비교

| 모델 | 파라미터 수 | 주요 연산 | Pruning 난이도 | Quantization 난이도 |
|------|------------|----------|---------------|-------------------|
| **MobileNetV2** | **~2.3M** | Depthwise Sep Conv | **쉬움** | **쉬움** |
| VGG16 | ~15M | 표준 3x3 Conv | 중간 | 중간 |
| ResNet18 | ~11M | 표준 Conv + Skip | 중간 | 어려움 (Skip 문제) |
| DenseNet121 | ~7M | Dense Connection | 어려움 | 매우 어려움 |
| EfficientNet-B0 | ~5M | SE + Swish | 어려움 | 매우 어려움 |

### 2.1 VGG

```python
# 단순 스택 구조
Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU → MaxPool → ...
```

**장점:** 구조가 단순해서 분석 용이
**단점:**
- 파라미터가 너무 많음 (FC layer에 집중)
- Pruning 후 정확도 급락 가능성

### 2.2 ResNet

```python
# Skip Connection 문제
out = F.relu(out + shortcut(x))  # 덧셈 후 ReLU
```

**Quantization 문제점:**
- Skip connection의 덧셈 연산에서 스케일 불일치 발생
- 서로 다른 양자화 스케일을 가진 텐서의 덧셈 → 정밀도 손실
- `FloatFunctional.add()` 같은 특수 처리 필요

### 2.3 DenseNet

```python
# Dense Connection
out = torch.cat([out, x], 1)  # 채널 방향 concat
```

**Pruning/Quantization 문제점:**
- 모든 이전 레이어 출력이 연결됨 → 하나 제거 시 전체 영향
- Concat 연산의 양자화 스케일 통합이 복잡
- 메모리 사용량이 높아 모바일 배포 부적합

### 2.4 EfficientNet

```python
# Swish 활성화 함수
def swish(x):
    return x * x.sigmoid()  # x * σ(x)

# SE (Squeeze-and-Excitation) Block
out = F.adaptive_avg_pool2d(x, 1)  # Global pooling
out = swish(self.se1(out))
out = self.se2(out).sigmoid()
out = x * out  # Channel-wise attention
```

**Quantization 문제점:**
- Swish: 비선형 함수의 양자화가 어려움 (ReLU처럼 단순하지 않음)
- Sigmoid: [0,1] 범위지만 미세한 값 차이가 중요 → 정밀도 손실
- SE block의 곱셈: 동적 스케일링 → 양자화 범위 예측 어려움
- 라이브러리 없이 구현 시 매우 복잡

---

## 3. MobileNetV2가 Pruning에 적합한 이유

### 3.1 Expansion Layer의 여유 공간

```python
cfg = [(1,  16, 1, 1),
       (6,  24, 2, 1),   # expansion=6 → 6배 확장
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       ...]
```

- Expansion ratio 6 = 입력 채널의 6배로 확장
- 이 확장된 공간에서 **중복된 필터가 많이 존재**
- Pruning으로 50-70% 제거해도 핵심 정보 유지 가능

### 3.2 Depthwise Conv의 독립성

```python
# groups=planes → 각 채널이 독립적으로 연산
self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                       stride=stride, padding=1, groups=planes)
```

- 채널 간 의존성 없음
- 특정 채널 제거 시 다른 채널에 영향 없음
- **구조적 Pruning(Structured Pruning)에 이상적**

### 3.3 L1 Norm 기반 Pruning 효과

Depthwise conv의 각 필터는 3x3=9개 파라미터만 가짐:
- 필터 중요도 판단이 명확
- L1 norm이 작은 필터 = 출력 기여도 낮음
- 과감한 pruning 가능

---

## 4. MobileNetV2가 Quantization에 적합한 이유

### 4.1 활성화 함수의 범위 제한

```python
# ReLU6: [0, 6] 범위
F.relu(self.bn1(self.conv1(x)))  # 실제로는 ReLU6 사용 권장
```

- 출력 범위가 명확 → 양자화 스케일 계산 용이
- 클리핑 불필요 → 정보 손실 최소화
- INT8로 [0, 6]을 256단계로 표현 → 충분한 해상도

### 4.2 BN-Conv 패턴의 Fusion

```python
# 학습 시
x → Conv → BN → ReLU

# 추론 시 (Fused)
x → FusedConvBNReLU  # 하나의 연산으로 통합
```

MobileNetV2의 깔끔한 구조:
- `Conv → BN → ReLU` 패턴이 일관됨
- Batch Normalization folding 적용 용이
- 연산 수 감소 + 양자화 포인트 감소

### 4.3 Linear Bottleneck의 대칭성

```python
# Projection layer (ReLU 없음)
out = self.bn3(self.conv3(out))  # Linear output
```

- 출력이 음수/양수 모두 포함
- **대칭 양자화(Symmetric Quantization)** 적용 가능
- Zero-point = 0으로 설정 가능 → 연산 단순화

### 4.4 Skip Connection의 안전성

```python
# MobileNetV2의 skip connection
out = out + self.shortcut(x) if self.stride==1 else out
```

- Shortcut이 identity 또는 1x1 conv (단순)
- ResNet처럼 복잡한 다운샘플링 없음
- 스케일 매칭이 상대적으로 용이

---

## 5. Pruning & Quantization 전략 제안

### 5.1 권장 파이프라인

```
[1] Baseline Training
         ↓
[2] Pruning (Unstructured L1)
         ↓
[3] Fine-tuning (성능 복구)
         ↓
[4] Quantization (PTQ or QAT)
         ↓
[5] Final Evaluation
```

### 5.2 Pruning 전략

#### Phase 1: Sensitivity Analysis
```python
# 각 pruning ratio에서 정확도 측정
ratios = [0.3, 0.5, 0.6, 0.7, 0.8]
for ratio in ratios:
    pruned_model = prune(model, ratio)
    acc = evaluate(pruned_model)
    print(f"Ratio {ratio}: Acc {acc}")
```

**목표:** 정확도 하락 < 2%인 최대 pruning ratio 찾기

#### Phase 2: Global Unstructured Pruning
```python
import torch.nn.utils.prune as prune

parameters_to_prune = []
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.7,  # 70% 제거
)
```

**왜 Global Unstructured?**
- 레이어별로 중요도가 다름
- Global: 전체에서 가장 덜 중요한 70% 제거
- 초기 레이어는 적게, 후반 레이어는 많이 pruning됨

#### Phase 3: Fine-tuning
```python
# 낮은 learning rate로 재학습
optimizer = optim.SGD(model.parameters(), lr=0.001)
for epoch in range(5):
    train(model)
```

### 5.3 Quantization 전략

#### Option A: Post-Training Quantization (PTQ)
```python
# 간단하지만 정확도 손실 가능
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibration (대표 데이터로 통계 수집)
with torch.no_grad():
    for inputs, _ in calibration_loader:
        model(inputs)

torch.quantization.convert(model, inplace=True)
```

#### Option B: Quantization-Aware Training (QAT)
```python
# 더 높은 정확도, 추가 학습 필요
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# Fake quantization으로 학습
for epoch in range(10):
    train(model)

torch.quantization.convert(model, inplace=True)
```

**MobileNetV2 권장:** PTQ로 시작 → 정확도 부족 시 QAT

### 5.4 예상 결과

| 단계 | 모델 크기 | 정확도 | 비고 |
|------|----------|--------|------|
| Baseline | 9.0 MB | 94.5% | FP32 |
| Pruned (70%) | 9.0 MB (sparse) | 93.0% | 실제 크기 동일 |
| Fine-tuned | 9.0 MB | 94.0% | 성능 복구 |
| Quantized (INT8) | **2.3 MB** | 93.5% | **~4x 압축** |

---

## 6. 라이브러리 없이 구현 시 주의사항

### 6.1 Pruning 구현
```python
# PyTorch 기본 prune 모듈만 사용
import torch.nn.utils.prune as prune

# 마스크 영구 적용 (sparse tensor로 변환 안 함)
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')
```

### 6.2 Quantization 구현
```python
# PyTorch 기본 quantization만 사용
import torch.quantization

# MobileNetV2 특화 wrapper 필요
class QuantizedMobileNetV2(nn.Module):
    def __init__(self, model_fp32):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x
```

### 6.3 피해야 할 복잡한 기법들
- Neural Architecture Search (NAS) 기반 pruning
- Knowledge Distillation (별도 teacher 모델 필요)
- Mixed-precision quantization (레이어별 다른 bit-width)
- Hardware-aware optimization (특정 하드웨어 종속)

---

## 7. 결론

**MobileNetV2가 Pruning & Quantization에 최적인 이유:**

1. **구조적 단순함**: Depthwise Separable Conv + Linear Bottleneck
2. **여유 있는 용량**: Expansion ratio 6으로 pruning 공간 확보
3. **범위 제한**: ReLU6로 양자화 친화적 출력
4. **독립적 채널**: Depthwise conv로 구조적 pruning 용이
5. **깔끔한 패턴**: BN fusion, skip connection 처리 용이

**vs 다른 모델:**
- VGG: 너무 큼, FC layer 문제
- ResNet: Skip connection 양자화 어려움
- DenseNet: Dense connection으로 pruning 불가
- EfficientNet: Swish, SE block 양자화 복잡

**권장 전략:**
```
Train → Prune 70% → Fine-tune 5 epochs → PTQ (INT8)
예상 결과: 모델 크기 4배 감소, 정확도 1% 이내 손실
```
