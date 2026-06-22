"""Category-GRU 모델 코드 리뷰 복사본.

원본: src/models/next_category/gru_model.py
판정: 현재 model.pt, model_config.json과 구조가 일치하며 수정이 필요한 오류 없음.
주의: 이 파일은 설명용 복사본이며 실제 학습/서빙은 원본 모듈을 사용한다.
"""
import torch
from torch import nn


class CategoryGRU(nn.Module):
    """카테고리 시퀀스와 이벤트 수치 피처를 결합하는 다음 카테고리 분류기."""

    def __init__(
        self,
        *,
        vocab_size,
        numeric_size=6,
        embedding_dim=64,
        numeric_dim=16,
        hidden_size=128,
        num_layers=1,
        dropout=0.2,
    ):
        super().__init__()

        # [리뷰] category_id를 그대로 연산하지 않고 밀집 벡터로 변환한다.
        # padding_idx=0은 패딩 토큰의 임베딩을 0으로 유지해 실제 카테고리와 구분한다.
        # 저장 모델의 shape은 (516, 64)로 현재 설정과 일치한다.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # [리뷰] 이벤트 종류 One-hot 4개 + log 시간 간격 + log 가격, 총 6개 피처를
        # 16차원으로 투영한다. ReLU는 수치 피처의 비선형 조합을 학습하게 한다.
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_size, numeric_dim),
            nn.ReLU(),
        )

        # [리뷰] 각 시점의 입력 크기는 category embedding과 numeric projection의 합이다.
        # batch_first=True이므로 입력 계약은 (batch, sequence, feature)다.
        # num_layers=1일 때 PyTorch GRU 내부 dropout은 적용되지 않아 아래의 별도 Dropout을 쓴다.
        self.gru = nn.GRU(
            embedding_dim + numeric_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # [리뷰] 마지막 Hidden State가 특정 패턴에 과도하게 의존하는 것을 줄이는 규제다.
        self.dropout = nn.Dropout(dropout)

        # [리뷰] 마지막 세션 상태 128차원을 전체 카테고리 Logit으로 변환한다.
        # 저장 모델의 출력 shape은 (516, 128)로 model_config와 일치한다.
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_cat, x_num, lengths):
        """최근 이벤트 시퀀스를 읽고 다음 카테고리별 Logit을 반환한다.

        x_cat: (B, T), 카테고리 인덱스
        x_num: (B, T, 6), 이벤트 수치 피처
        lengths: (B,), 패딩을 제외한 실제 시퀀스 길이
        """
        category = self.embedding(x_cat)
        numeric = self.numeric_projection(x_num)

        # [리뷰] 이벤트마다 카테고리 의미와 행동·가격·시간 정보를 하나의 벡터로 결합한다.
        sequence = torch.cat([category, numeric], dim=-1)

        # [리뷰] pack_padded_sequence를 사용해 GRU가 오른쪽 패딩을 실제 이벤트로 읽지 않는다.
        # enforce_sorted=False이므로 DataLoader에서 길이순 정렬을 별도로 할 필요가 없다.
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # [리뷰] 전체 시점 출력은 필요하지 않고 마지막 레이어의 Hidden State만 사용한다.
        _, hidden = self.gru(packed)
        logits = self.output(self.dropout(hidden[-1]))

        # [리뷰] index 0은 padding 전용이므로 추천 후보가 되지 못하게 최솟값으로 마스킹한다.
        # Unknown index 1은 현재 평가 Top-10에 한 번도 등장하지 않았고, 실제 inference에서는
        # padding과 unknown을 모두 다시 제외한다. 현재 동작에는 문제없다.
        logits[:, 0] = torch.finfo(logits.dtype).min
        return logits
