"""
Environment Setup Module

Tự động phát hiện và cấu hình môi trường cho notebook chạy trên:
- Local (conda environment)
- Google Colab
- Kaggle Kernels

Usage trong notebook:
    from core import setup_env
    env = setup_env.setup()
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Literal


class KaggleEnvironment:
    """Quản lý môi trường cho Kaggle competitions"""

    def __init__(self):
        self.env_type = self._detect_env()
        self.root_path = self._get_root_path()
        self.data_path = self._get_data_path()

    def _detect_env(self) -> Literal["local", "colab", "kaggle"]:
        """Phát hiện môi trường hiện tại"""
        if "google.colab" in sys.modules:
            return "colab"
        elif "kaggle_web_client" in sys.modules or os.path.exists("/kaggle"):
            return "kaggle"
        else:
            return "local"

    def _get_root_path(self) -> Path:
        """Lấy đường dẫn root của project"""
        if self.env_type == "colab":
            # Giả sử repo được clone vào MyDrive/kaggle-competitions
            return Path("/content/drive/MyDrive/kaggle-competitions")
        elif self.env_type == "kaggle":
            return Path("/kaggle/working")
        else:
            # Local: Lấy thư mục cha của core/
            return Path(__file__).resolve().parent.parent

    def _get_data_path(self) -> Path:
        """Lấy đường dẫn data cho từng môi trường"""
        if self.env_type == "colab":
            # Data trong Drive hoặc uploaded
            return Path("/content")
        elif self.env_type == "kaggle":
            # Kaggle input path
            return Path("/kaggle/input")
        else:
            # Local: competitions/[competition]/data
            return self.root_path / "competitions"

    def setup(
        self, repo_user: str = "n24q02m", repo_name: str = "n24q02m-kaggle-competitions"
    ):
        """
        Thiết lập môi trường

        Args:
            repo_user: GitHub username (dùng cho cloud)
            repo_name: Repository name (dùng cho cloud)
        """
        print(f"🚀 Detected Environment: {self.env_type.upper()}")
        print(f"📂 Root Path: {self.root_path}")
        print(f"📊 Data Path: {self.data_path}")

        # Setup theo môi trường
        if self.env_type == "colab":
            self._setup_colab()
            self._install_requirements(repo_user, repo_name)
        elif self.env_type == "kaggle":
            self._setup_kaggle()
            self._install_requirements(repo_user, repo_name)
        else:
            self._setup_local()

        print("✅ Setup Complete!")
        return self

    def _setup_local(self):
        """Setup cho local environment"""
        # Add root path to system path để import được core modules
        if str(self.root_path) not in sys.path:
            sys.path.insert(0, str(self.root_path))
        print("ℹ️  Local: Sử dụng conda environment. Không tự động cài thư viện.")
        print("   Chạy: conda env create -f environment.yml (nếu chưa tạo env)")
        print("   Chạy: conda activate kaggle-competitions")

    def _setup_colab(self):
        """Setup cho Google Colab"""
        # Mount Google Drive nếu chưa
        if not os.path.exists("/content/drive"):
            try:
                from google.colab import drive

                print("📌 Mounting Google Drive...")
                drive.mount("/content/drive")
            except Exception as e:
                print(f"⚠️  Warning: Could not mount Drive: {e}")

    def _setup_kaggle(self):
        """Setup cho Kaggle Kernels"""
        print("ℹ️  Kaggle: Environment sẵn sàng")

    def _install_requirements(self, repo_user: str, repo_name: str):
        """
        Cài đặt requirements từ GitHub raw

        Args:
            repo_user: GitHub username
            repo_name: Repository name
        """
        try:
            import requests

            # Download requirements.txt từ GitHub
            url = f"https://raw.githubusercontent.com/{repo_user}/{repo_name}/main/requirements.txt"
            print(f"📦 Downloading requirements from: {url}")

            response = requests.get(url)
            response.raise_for_status()

            # Save requirements.txt tạm
            req_file = Path("requirements.txt")
            req_file.write_text(response.text)

            # Install requirements
            print("⏳ Installing requirements...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
                stderr=subprocess.DEVNULL,
            )
            print("✅ Requirements installed")

            # Cleanup
            req_file.unlink()

        except Exception as e:
            print(f"⚠️  Warning: Could not install requirements: {e}")
            print("   Tiếp tục với các thư viện có sẵn...")

    def info(self):
        """In thông tin môi trường"""
        print("\n" + "=" * 50)
        print("ENVIRONMENT INFO")
        print("=" * 50)
        print(f"Type: {self.env_type}")
        print(f"Root: {self.root_path}")
        print(f"Data: {self.data_path}")
        print("=" * 50 + "\n")


def setup(
    repo_user: str = "n24q02m", repo_name: str = "n24q02m-kaggle-competitions"
) -> KaggleEnvironment:
    """
    Quick setup function - wrapper around KaggleEnvironment

    Args:
        repo_user: GitHub username
        repo_name: Repository name

    Returns:
        Configured KaggleEnvironment instance

    Example:
        >>> from core import setup_env
        >>> env = setup_env.setup()
        >>> print(env.env_type)
        'local'
    """
    env = KaggleEnvironment()
    env.setup(repo_user, repo_name)
    return env


if __name__ == "__main__":
    # Test
    env = setup()
    env.info()
