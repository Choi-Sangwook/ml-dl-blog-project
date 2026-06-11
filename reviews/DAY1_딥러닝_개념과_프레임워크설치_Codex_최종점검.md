# DAY1 딥러닝 개념과 프레임워크 설치 - Codex 최종점검

## 게시 가능 여부

**판정: 필수 문제 수정 후 게시 가능**

기존 Codex 통합검토안의 주요 정확성·안전성 지적은 최종본에 대부분
반영되었다. 다만 아래 4건은 게시 전에 반드시 수정해야 한다.

## 반드시 수정할 문제

### 1. Transformer 연도 수정

**위치:** 역사 연표 및 핵심 정리

최종본의 `Transformer(2020~)` 표기는 부정확하다. Transformer 논문은
2017년에 발표되었으며, ChatGPT 공개 시점과도 구분해야 한다.

```text
Transformer(2017) → ChatGPT(2022)
```

### 2. 설치 명령에 `matplotlib` 추가

**위치:** 심장병 분류 전체 실행 예제의 설치 주석

본문의 학습 곡선 코드가 `matplotlib.pyplot`을 사용하지만 설치 명령에는
`matplotlib`이 빠져 있다.

```bash
pip install tensorflow scikit-learn ucimlrepo matplotlib
```

TensorFlow 설치 가능 여부는 Python 버전과 운영체제에 따라 달라질 수
있으므로, 설치 전에 공식 지원 버전을 확인한다는 안내도 함께 넣어야 한다.

- [TensorFlow 공식 설치 문서](https://www.tensorflow.org/install/pip)

### 3. Keras 비교표의 열과 값 일치

**위치:** `6.1 대표 프레임워크` 표

표의 열 이름은 `개발`인데 Keras 행에는 개발 주체가 아니라
`(현재는 멀티 백엔드)`라는 특징이 들어가 있다. 다음 중 하나로 고쳐야 한다.

- Keras 행의 개발 주체를 적고 멀티 백엔드 내용은 `특징` 열로 이동
- `개발` 열을 `개발·생태계`처럼 실제 내용에 맞는 이름으로 변경

Keras 3가 TensorFlow·JAX·PyTorch 백엔드를 지원한다는 본문 설명 자체는
유지해도 된다.

- [Keras 3 공식 소개](https://keras.io/keras_3/)

### 4. 보스턴 집값 참고 링크 교체

**위치:** 참고 자료

현재 연결된 stable 버전의 `load_boston` 문서 주소는 존재하지 않는다.
제거 이유와 윤리 문제를 확인할 수 있는 보존된 문서로 교체해야 한다.

- [scikit-learn 1.1 `load_boston` 문서](https://scikit-learn.org/1.1/modules/generated/sklearn.datasets.load_boston.html)

## 최종 확인 결과

- PDF 52쪽 전체를 확인했으며, 텍스트가 거의 없는 24~52쪽은 삽입 이미지를
  페이지별로 점검했다.
- Python 코드 블록 5개는 문법 파싱을 통과했다.
- Markdown 코드 펜스는 정상적으로 닫혀 있다.
- 현재 검토 환경에는 TensorFlow, PyTorch, scikit-learn, ucimlrepo가 없어
  전체 학습 코드는 실제 실행하지 못했다.
- 요청에 적힌 PDF 경로는 존재하지 않았으며, 실제 대응 파일인
  `sources/DAY1_딥러닝개념_프레임워크설치.pdf`를 기준으로 대조했다.

위 4건을 반영하면 게시 가능한 상태다.
