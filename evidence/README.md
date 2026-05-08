# Evidence: Prompt V1 vs V2

## RAGAS Scores

| Metric | V1 | V2 |
|--------|----|----|
| Faithfulness | 0.869 | 0.803 |
| Answer Relevancy | 0.903 | 0.869 |
| Context Recall | 1.000 | 1.000 |
| Context Precision | 0.958 | 0.952 |

## Result
**V1 wins** on faithfulness (+6.6%), relevancy (+3.8%), and precision (+0.6%). Both achieve perfect context recall.

## A/B Routing
- V1: 19 questions (38%)
- V2: 31 questions (62%)

## Conclusion
V1 is better for quality-focused use cases. V2 can handle higher volume despite lower scores. Recommend V1 for production baseline.
