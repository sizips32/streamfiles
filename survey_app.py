# -*- coding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import platform
import os

def load_korean_font():
    """환경에 맞는 한글 폰트를 설정합니다."""
    if platform.system() == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif platform.system() == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux 또는 기타
        font_dirs = ['/usr/share/fonts/truetype/nanum/', '.']
        font_files = fm.findSystemFonts(fontpaths=font_dirs)
        font_path = next((file for file in font_files if 'NanumGothic' in file), None)
        
        if not font_path:
            # 폰트를 다운로드
            font_url = "https://github.com/googlefonts/nanum-gothic/blob/main/fonts/NanumGothic-Regular.ttf?raw=true"
            font_path = "NanumGothic.ttf"
            if not os.path.exists(font_path):
                import urllib.request
                urllib.request.urlretrieve(font_url, font_path)
        
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False

traits = {
    "유다": {
        "성품": "리더십과 용기",
        "본성": "타락과 탐욕 (창 38장 - 다말과의 부적절한 관계)",
        "성경구절": "창 49:9",
        "대표인물": "다윗,솔로몬,갈렙",
        "선교사명": "찬양은 전도와 선교의 문을 여는 열쇠입니다. 믿음의 리더들은 찬양과 경배를 통해 하나님의 능력을 드러내며 복음을 전해야 합니다."
    },
    "르우벤": {
        "성품": "열정과 회복",
        "본성": "충동적 행동과 불안정성 (창 49:4 - 빌하와의 관계로 인해 맏아들의 권리를 상실함)",
        "성경구절": "창 49:3",
        "대표인물": "르우벤",
        "선교사명": "선교사와 전도자는 실수를 통해 겸손을 배우고 회복의 길을 걷습니다. 실패를 두려워하지 말고 하나님의 용서와 은혜를 나누며 사역해야 합니다"
    },
    "스불론": {
        "성품": "사교성과 나눔",
        "본성": "세속적 유혹에 취약 (이방 문화에 쉽게 동화될 가능성)",
        "성경구절": "창 49:13",
        "대표인물": "사사 엘론",
        "선교사명": "선교사는 열방을 향한 비전을 품고 복음을 전하며 세상과 연결되어야 합니다."
    },
    "잇사갈": {
        "성품": "성실함과 감사",
        "본성": "지나친 타협과 안일함 (창 49:15 - 편안함을 위해 종이 되는 선택)",
        "성경구절": "창 49:14",
        "대표인물": "바락",
        "선교사명": "선교사는 맡은 사명을 지혜롭고 성실하게 감당해야 합니다."
    },
    "갓": {
        "성품": "믿음과 영적 전쟁",
        "본성": "끊임없는 갈등 (전쟁의 삶으로 인해 평화가 부족함)",
        "성경구절": "창 49:19",
        "대표인물": "엘리야",
        "선교사명": "영적 전쟁에서 물러서지 않는 자세는 전도자의 필수 덕목입니다. 기도로 무장하고 끝까지 복음을 전파해야 합니다."
    },
    "아셀": {
        "성품": "섬김과 기쁨",
        "본성": "나태함과 안락함 추구 (창 49:20 - 물질적 풍요에 지나치게 의존할 가능성)",
        "성경구절": "창 49:20",
        "대표인물": "안나 여선지자",
        "선교사명": "선교사는 하나님이 주신 풍성함을 나누는 사람들입니다. 물질적, 영적 은혜를 나누며 이웃을 섬겨야 합니다."
    },
    "납달리": {
        "성품": "소통과 온화함",
        "본성": "우유부단함 (결정 상황에서 흔들릴 가능성)",
        "성경구절": "창 49:21",
        "대표인물": "바락",
        "선교사명": "전도자는 진리의 말씀을 아름답게 전하며 영혼을 자유케 하는 사명을 감당해야 합니다."
    },
    "레위": {
        "성품": "신앙과 순종",
        "본성": "잔혹함과 과도한 열정 (창 49:5-7 - 세겜 사건에서의 폭력적인 복수)",
        "성경구절": "창 49:7",
        "대표인물": "모세, 아론",
        "선교사명": "거룩함과 헌신은 선교자의 중심 덕목입니다. 하나님의 임재를 전하는 제사장 역할을 감당해야 합니다."
    },
    "요셉": {
        "성품": "긍정적 영향과 나눔",
        "본성": "과도한 이상주의 (현실적 문제를 간과할 가능성)",
        "성경구절": "창 49:22",
        "대표인물": "여호수아, 드보라",
        "선교사명": "용서와 사랑은 복음을 전하는 핵심입니다. 어려움 속에서도 하나님의 형통함을 나누어야 합니다."
    },
    "베냐민": {
        "성품": "끈기와 열정",
        "본성": "파괴적 열정 (창 49:27 - 전투와 경쟁심이 지나칠 가능성)",
        "성경구절": "창 49:27",
        "대표인물": "사도 바울",
        "선교사명": "전도자는 담대한 믿음으로 어려움을 극복하며 복음을 전파해야 합니다."
    },
    "시므온": {
        "성품": "열정적 복음 전파",
        "본성": "폭력적 성향 (창 49:5 - 세겜 사건에서의 과도한 복수)",
        "성경구절": "창 49:7",
        "대표인물": "시므온",
        "선교사명": "전도자는 하나님의 말씀에 귀를 기울이며 인내로 사람들의 마음을 얻어야 합니다."
    },
    "므낫세": {
        "성품": "장기 계획과 충성",
        "본성": "강한 독립심으로 인한 분열 가능성 (공동체 조화를 깨뜨릴 위험)",
        "성경구절": "창 48:19",
        "대표인물": "기드온, 야일",
        "선교사명": "선교는 상처받은 영혼에게 새로운 소망을 주는 사역입니다. 과거의 상처를 치유하며 하나님의 새 일을 전해야 합니다."
    },
}

def display_swot_in_sidebar(top_tribes, bottom_tribes):
    """SWOT 분석 해당 지파를 사이드바 최상단에 상세히 표시"""
    st.sidebar.title("SWOT 지파 특성 상세보기")

    # 추가적인 안내 문구
    st.sidebar.write("아래는 이번 설문 결과를 바탕으로 선정된 네 지파의 특징입니다.")
    st.sidebar.write("각 지파별로 성품, 본성, 관련 성경구절 및 대표인물과 선교사명을 보여줍니다.\n")

    # 강점 지파
    st.sidebar.subheader("강점 (Strength)")
    top_str = top_tribes[0][0]
    st.sidebar.markdown(f"**지파명:** {top_str}")
    st.sidebar.write(f"**성품:** {traits[top_str]['성품']}")
    st.sidebar.write(f"**본성:** {traits[top_str]['본성']}")
    st.sidebar.write(f"**성경구절:** {traits[top_str]['성경구절']}")
    st.sidebar.write(f"**대표인물:** {traits[top_str]['대표인물']}")
    st.sidebar.write(f"**선교사명:** {traits[top_str]['선교사명']}")

    # 기회 지파
    st.sidebar.subheader("기회 (Opportunity)")
    top_opp = top_tribes[1][0]
    st.sidebar.markdown(f"**지파명:** {top_opp}")
    st.sidebar.write(f"**성품:** {traits[top_opp]['성품']}")
    st.sidebar.write(f"**본성:** {traits[top_opp]['본성']}")
    st.sidebar.write(f"**성경구절:** {traits[top_opp]['성경구절']}")
    st.sidebar.write(f"**대표인물:** {traits[top_opp]['대표인물']}")
    st.sidebar.write(f"**선교사명:** {traits[top_opp]['선교사명']}")

    # 약점 지파명
    st.sidebar.subheader("약점 (Weakness)")
    bottom_weak = bottom_tribes[0][0]
    st.sidebar.markdown(f"**지파명:** {bottom_weak}")
    st.sidebar.write(f"**성품:** {traits[bottom_weak]['성품']}")
    st.sidebar.write(f"**본성:** {traits[bottom_weak]['본성']}")
    st.sidebar.write(f"**성경구절:** {traits[bottom_weak]['성경구절']}")
    st.sidebar.write(f"**대표인물:** {traits[bottom_weak]['대표인물']}")
    st.sidebar.write(f"**선교사명:** {traits[bottom_weak]['선교사명']}")

    # 위협 지파명
    st.sidebar.subheader("위협 (Threat)")
    bottom_threat = bottom_tribes[1][0]
    st.sidebar.markdown(f"**지파명:** {bottom_threat}")
    st.sidebar.write(f"**성품:** {traits[bottom_threat]['성품']}")
    st.sidebar.write(f"**본성:** {traits[bottom_threat]['본성']}")
    st.sidebar.write(f"**성경구절:** {traits[bottom_threat]['성경구절']}")
    st.sidebar.write(f"**대표인물:** {traits[bottom_threat]['대표인물']}")
    st.sidebar.write(f"**선교사명:** {traits[bottom_threat]['선교사명']}")

def plot_survey_results(scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    tribes = list(scores.keys())
    values = list(scores.values())

    ax.bar(tribes, values, color="skyblue")
    ax.set_title("설문조사 결과", fontsize=16)
    ax.set_xlabel("지파", fontsize=12)
    ax.set_ylabel("점수 합계", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def display_swot_analysis_graph(top_tribes, bottom_tribes):
    strengths_map = {
        "유다": "리더십과 용기", "갓": "믿음과 영적 전쟁", "베냐민": "끈기와 열정",
        "르우벤": "열정과 회복", "시므온": "열정적 복음 전파", "납달리": "소통과 온화함",
        "스불론": "사교성과 나눔", "아셀": "섬김과 기쁨", "요셉": "긍정적 영향과 나눔",
        "잇사갈": "성실함과 감사", "레위": "신앙과 순종", "므낫세": "장기 계획과 충성"
    }
    
    swot_data = {
        "강점 (Strengths)": f"{top_tribes[0][0]}: {strengths_map[top_tribes[0][0]]}\n이(가) 빛납니다.",
        "기회 (Opportunities)": f"{top_tribes[1][0]}: {strengths_map[top_tribes[1][0]]}\n이(가) 돋보입니다.",
        "약점 (Weaknesses)": f"{bottom_tribes[0][0]}: {strengths_map[bottom_tribes[0][0]]}\n을(를) 더욱 발전시키면 좋겠습니다.",
        "위협 (Threats)": f"{bottom_tribes[1][0]}: {strengths_map[bottom_tribes[1][0]]}\n이(가) 많이 부족합니다.",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.add_patch(Rectangle((0, 0.5), 0.5, 0.5, edgecolor='black', facecolor='lightgreen', lw=2))
    ax.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, edgecolor='black', facecolor='lightcoral', lw=2))
    ax.add_patch(Rectangle((0, 0), 0.5, 0.5, edgecolor='black', facecolor='lightskyblue', lw=2))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.5, edgecolor='black', facecolor='lightyellow', lw=2))

    ax.text(0.25, 0.75, f"강점\n{swot_data['강점 (Strengths)']}", ha='center', va='center', fontsize=10)
    ax.text(0.75, 0.75, f"약점\n{swot_data['약점 (Weaknesses)']}", ha='center', va='center', fontsize=10)
    ax.text(0.25, 0.25, f"기회\n{swot_data['기회 (Opportunities)']}", ha='center', va='center', fontsize=10)
    ax.text(0.75, 0.25, f"위협\n{swot_data['위협 (Threats)']}", ha='center', va='center', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    st.pyplot(fig)

def calculate_results(scores):
    averages = {tribe: total / 2 for tribe, total in scores.items()}
    sorted_tribes = sorted(averages.items(), key=lambda item: item[1], reverse=True)
    top_tribes = sorted_tribes[:2]
    bottom_tribes = sorted_tribes[-2:]

    st.subheader("설문 결과")
    st.write("최고점 및 최저점 지파 결과를 보여줍니다:")
    for tribe, average in top_tribes:
        st.write(f"**최고점 지파 - {tribe}:** 평균 {average:.2f}점")
    for tribe, average in bottom_tribes:
        st.write(f"**최저점 지파 - {tribe}:** 평균 {average:.2f}점")

    # 사이드바 최상단에 SWOT 해당 지파의 특성 상세 표시
    display_swot_in_sidebar(top_tribes, bottom_tribes)

    # 설문조사 결과 그래프 출력
    plot_survey_results(scores)

    # SWOT 분석 그래프 출력
    display_swot_analysis_graph(top_tribes, bottom_tribes)

def conduct_survey():
    st.title("열두 지파 성품 설문조사")
    st.write("각 항목에 대해 1에서 5점 사이로 점수를 입력해주세요.")
    
    questions = {
        "유다": [
            "나는 위기 상황이 닥쳤을 때, 주저하지 않고 앞장서서 해결하려는 경향이 있다.",
            "나는 어려움에 처한 사람 보면, 기꺼이 나를 희생하여 돕고자 한다.",
        ],
        "르우벤": [
            "나는 새로운 일을 시작할 때, 쉽게 의욕이 넘치고 정적으로 임하는 편이다.",
            "나는 과거의 잘못을 돌아보고, 그것으로부터 교훈을 얻어 성장하려 노력한다.",
        ],
        "스불론": [
            "나는 낯선 사람들과도 스스럼없이 어울리고, 그들과 함께 하는 것을 즐긴다.",
            "나는 가지고 있는 자원을 효율적으로 관리하고, 필요한 사람들과 나누는 것에 보람을 느낀다.",
        ],
        "잇사갈": [
            "나는 한 번 맡은 일은 어려움이 있더라도 끝까지 완수하는 편이다.",
            "나는 현재 내가 가진 것들에 감사하고, 주어진 상황에 만족하며 살아간다.",
        ],
        "갓": [
            "나는 힘든 상황 속에서도 나의 믿음을 굳건히 지켜나가려 애쓴다.",
            "나는 영적인 성장을 위해 꾸준히 기도하며, 어려움을 극복하려 한다.",
        ],
        "아셀": [
            "나는 다른 사람을 돕는 일에서 큰 기쁨과 보람을 느낀다.",
            "나는 나의 영적인 은사를 다른 사람들과 나누며, 그들을 격려하는 것을 좋아한다.",
        ],
        "납달리": [
            "나는 나의 생각과 의견을 상대방에게 명확하고 효과적으로 전달하는 편이다.",
            "나는 상대방과 소통할 때, 부드럽고 온화한 태도를 유지하려 노력한다.",
        ],
        "레위": [
            "나는 하나님의 말씀을 듣고, 그 가르침에 따라 살아가려고 노력한다.",
            "나는 과거에 잘못한 일이 있다면, 그것을 인정하고 바로잡으려 애쓴다.",
        ],
        "요셉": [
            "나는 주변 사람들에게 긍정적인 영향을 미치며, 그들의 삶에 좋은 본보기가 되고자 한다.",
            "나는 영적,물질적으로 도움이 필요한 사람들에게 베풀고 나누는 삶을 실천한다.",
        ],
        "베냐민": [
            "나는 사람들에게 복음을 전하는 일에 적극적으로 참여하고, 그들에게 하나님의 사랑을 알리고 싶다.",
            "나는 한 번 시작한 일은 어떤 어려움이 있어도 포기하지 않고 끝까지 해내는 끈기가 있다.",
        ],
        "시므온": [
            "나는 새로운 도전을 즐기며 받아들인다.",
            "나는 믿음을 나누는데 열정적이다.",
        ],
        "므낫세": [
            "나는 장기적인 계획을 세우고 실천한다.",
            "나는 맡은 자리에서 신실하게 섬긴다.",
        ],
    }

    scores = {tribe: 0 for tribe in questions}

    if 'show_results' not in st.session_state:
        st.session_state['show_results'] = False

    question_number = 1
    for tribe, tribe_questions in questions.items():
        for question in tribe_questions:
            scores[tribe] += st.slider(f"질문 {question_number}: {question}", 1, 5, 3)
            question_number += 1

    if st.button("결과 출력"):
        st.session_state['show_results'] = True
        calculate_results(scores)

    # 결과 출력 전까지 사이드바 비움
    if not st.session_state['show_results']:
        st.sidebar.empty()

if __name__ == "__main__":
    load_korean_font()
    conduct_survey()
