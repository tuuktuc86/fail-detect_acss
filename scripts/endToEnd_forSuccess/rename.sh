#!/bin/bash

# 기준 경로 (필요시 수정)
BASE_DIR="/AILAB-summer-school-2025/success_data_raw"

# 에피소드 번호 초기화
episode_number=1

# "len숫자_success"를 포함한 디렉토리를 이름순 정렬하여 반복
for dir in $(find "$BASE_DIR" -maxdepth 1 -type d -name "simulation_traj_*_len*_success" | sort); do
    # len 뒤 숫자 추출 (예: 474)
    step_count=$(echo "$dir" | sed -n 's/.*_len\([0-9]\+\)_success/\1/p')

    # 새 디렉토리 이름 만들기
    new_name="success_episode${episode_number}_steps${step_count}"

    # 절대 경로 처리
    current_path="$dir"
    new_path="$(dirname "$dir")/$new_name"

    echo "🔁 $current_path → $new_path"

    # 이름 바꾸기
    mv "$current_path" "$new_path"

    # 에피소드 번호 증가
    episode_number=$((episode_number + 1))
done
