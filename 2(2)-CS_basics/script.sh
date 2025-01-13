#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되지 않았습니다. 설치를 진행합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_install.sh
    bash miniconda_install.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Conda 초기화
source "$HOME/miniconda/etc/profile.d/conda.sh"

# Conda 환경 생성 및 활성화
if ! conda env list | grep -q 'myenv'; then
    conda create -y -n myenv python=3.9
fi
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    problem_number="${file%.py}"
    input_file="../input/${problem_number}_input"
    output_file="../output/${problem_number}_output"
    if [[ -f "$input_file" ]]; then
        python "$file" < "$input_file" > "$output_file"
        echo "${file} 실행 완료. 출력은 ${output_file}에 저장되었습니다."
    else
        echo "입력 파일 $input_file을 찾을 수 없습니다."
    fi
done

# mypy 테스트 실행
mypy *.py

# 가상환경 비활성화
conda deactivate
