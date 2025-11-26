# Kaggle Competitions Workspace

Workspace để lưu trữ, thực thi và quản lý các notebook cuộc thi Kaggle, hỗ trợ đa môi trường: **Local (conda)**, **Google Colab** và **Kaggle Kernels**.

## ✨ Tính Năng

- 🔄 **Multi-Environment Support**: Notebook tự động phát hiện và cấu hình môi trường
- 📦 **Reusable Core Utilities**: Thư viện dùng chung qua GitHub raw
- 🎯 **Competition Templates**: Template chuẩn cho mỗi cuộc thi
- 🛠️ **Local Development**: Conda environment và Makefile commands
- 📊 **Best Practices**: Quy trình ML workflow chuẩn

## 🚀 Quick Start

### Local Setup (Lần đầu)

```bash
# 1. Clone repo
git clone https://github.com/n24q02m/n24q02m-kaggle-competitions.git
cd n24q02m-kaggle-competitions

# 2. Tạo conda environment
make setup

# 3. Activate environment
conda activate kaggle-competitions

# 4. Chạy Jupyter Lab
make lab
```

### Google Colab

1. Mở notebook trên Colab
2. Chạy **Bootstrap Cell** (cell đầu tiên) - tự động setup
3. Bắt đầu code!

### Kaggle Kernels

1. Tạo notebook mới trên Kaggle
2. Copy **Bootstrap Cell** từ template
3. Chạy và code!

## 📁 Cấu Trúc Repo

```
n24q02m-kaggle-competitions/
├── core/                      # Utilities dùng chung
│   ├── __init__.py
│   └── setup_env.py          # Auto-detect environment
├── competitions/              # Các cuộc thi
│   └── titanic/
│       ├── README.md
│       ├── data/             # Data files (gitignored)
│       ├── notebooks/
│       │   └── solution.ipynb
│       ├── models/           # Saved models
│       └── submissions/      # Submission files
├── environment.yml           # Conda environment
├── requirements.txt          # Pip requirements
├── Makefile                  # Utility commands
├── README.md                 # This file
└── HANDBOOK.md              # Chi tiết hướng dẫn
```

## 📚 Competitions

| Competition                                 | Status        | Best Score | Notebook                                                        |
| ------------------------------------------- | ------------- | ---------- | --------------------------------------------------------------- |
| [Titanic](https://www.kaggle.com/c/titanic) | 🚧 In Progress | -          | [solution.ipynb](competitions/titanic/notebooks/solution.ipynb) |

## 🔧 Makefile Commands

```bash
make help         # Hiển thị tất cả commands
make setup        # Tạo conda environment
make install      # Cài/update dependencies
make lab          # Chạy Jupyter Lab
make clean        # Dọn dẹp cache
```

## 💡 Workflow

### Bắt đầu cuộc thi mới

1. **Tạo thư mục competition:**
   ```bash
   mkdir -p competitions/<competition-name>/{data,notebooks,models,submissions}
   ```

2. **Copy template notebook:**
   ```bash
   cp competitions/titanic/notebooks/solution.ipynb competitions/<competition-name>/notebooks/
   ```

3. **Update bootstrap cell** với đường dẫn đúng

4. **Download data** (xem hướng dẫn trong HANDBOOK.md)

5. **Code và submit!**

## 📖 Documentation

- [HANDBOOK.md](HANDBOOK.md) - Hướng dẫn chi tiết
- [Titanic README](competitions/titanic/README.md) - Hướng dẫn cuộc thi Titanic

## 🔗 Useful Links

- [Kaggle](https://www.kaggle.com/)
- [Kaggle API](https://github.com/Kaggle/kaggle-api)
- [Kaggle Learn](https://www.kaggle.com/learn)

## 🤝 Contributing

Đây là repo cá nhân cho mục đích học tập và thực hành.

## 📝 License

MIT License - Free to use and modify
