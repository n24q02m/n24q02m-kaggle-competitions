# Titanic - Machine Learning from Disaster

[Competition Link](https://www.kaggle.com/c/titanic)

## Mô Tả

Cuộc thi Titanic là bài toán nhập môn kinh điển trên Kaggle. Mục tiêu là dự đoán hành khách nào sống sót sau thảm họa chìm tàu Titanic dựa trên các đặc trưng như tuổi, giới tính, hạng vé, v.v.

## Dataset

### Train Data Columns

| Column | Description | Type |
|--------|-------------|------|
| `PassengerId` | ID hành khách | int |
| `Survived` | Sống sót (0 = No, 1 = Yes) | int |
| `Pclass` | Hạng vé (1 = 1st, 2 = 2nd, 3 = 3rd) | int |
| `Name` | Tên hành khách | string |
| `Sex` | Giới tính | string |
| `Age` | Tuổi | float |
| `SibSp` | Số anh chị em / vợ chồng trên tàu | int |
| `Parch` | Số cha mẹ / con cái trên tàu | int |
| `Ticket` | Số vé | string |
| `Fare` | Giá vé | float |
| `Cabin` | Số cabin | string |
| `Embarked` | Cảng lên tàu (C = Cherbourg, Q = Queenstown, S = Southampton) | string |

### Test Data

Giống train data nhưng không có cột `Survived`

## Evaluation Metric

**Accuracy**: Tỷ lệ dự đoán đúng

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

## Submission Format

File CSV với 2 cột:

```csv
PassengerId,Survived
892,0
893,1
894,0
...
```

## Download Data

### Sử dụng Kaggle API (Local)

```bash
# Setup Kaggle API credentials first
# Download ~/.kaggle/kaggle.json from Kaggle -> Account -> API -> Create New Token

# Download data
kaggle competitions download -c titanic -p competitions/titanic/data

# Unzip
cd competitions/titanic/data
unzip titanic.zip
```

### Manual Download

1. Truy cập [Titanic Competition Data](https://www.kaggle.com/c/titanic/data)
2. Download `train.csv`, `test.csv`, `gender_submission.csv`
3. Đặt vào `competitions/titanic/data/`

## Useful Resources

- [Titanic Tutorial - Alexis Cook](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)
- [EDA & Feature Engineering](https://www.kaggle.com/code/startupsci/titanic-data-science-solutions)
- [Simple Logistic Regression Baseline](https://www.kaggle.com/code/mnassrib/titanic-logistic-regression-with-python)

## Structure

```
titanic/
├── README.md              # This file
├── data/                  # Competition data (gitignored)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── solution.ipynb    # Main notebook
├── models/               # Saved models (gitignored)
└── submissions/          # Submission files
    └── .gitkeep
```

## Getting Started

1. **Download data** (xem hướng dẫn trên)
2. **Mở notebook**: `notebooks/solution.ipynb`
3. **Follow the workflow** trong notebook
4. **Submit** file từ `submissions/` lên Kaggle
