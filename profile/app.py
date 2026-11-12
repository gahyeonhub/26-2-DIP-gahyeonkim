import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as nd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="HW05 - Image Segmentation", layout="wide")
st.title("Object Counting")

# ---------------------------------------------------------------
# 1. 이미지 불러오기
#    (Colab의 drive.mount 대신 파일 업로드 사용)
#    업로드가 없으면 레포에 같이 올린 IMG.jpeg를 기본 이미지로 사용
# ---------------------------------------------------------------
uploaded = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
DEFAULT_IMG = os.path.join(os.path.dirname(__file__), "IMG.jpeg")

if uploaded is not None:
    I = Image.open(uploaded).convert("L")
elif os.path.exists(DEFAULT_IMG):
    I = Image.open(DEFAULT_IMG).convert("L")
else:
    st.info("왼쪽 사이드바에서 이미지를 업로드해 주세요.")
    st.stop()

f = np.array(I)

# 파라미터 (기본값은 원래 노트북 값과 동일)
st.sidebar.header("파라미터")
kernel = st.sidebar.slider("Median filter 크기", 3, 15, 7, step=2)
th_val = st.sidebar.slider("Threshold 값", 0, 255, 230)
iters = st.sidebar.slider("Closing / Opening 반복 횟수", 1, 10, 5)


def show_image(img, cmap="gray", colorbar=False, title=None):
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------
# 2. 원본 이미지
# ---------------------------------------------------------------
st.header("1. 원본 이미지 (Grayscale)")
show_image(f, colorbar=True)

# ---------------------------------------------------------------
# 3. Median filter로 노이즈 제거
# ---------------------------------------------------------------
st.header("2. Median Filter 적용")
f_denoised = signal.medfilt2d(f, kernel)
show_image(f_denoised, colorbar=True)

# ---------------------------------------------------------------
# 4. 히스토그램
# ---------------------------------------------------------------
st.header("3. 픽셀 밝기 히스토그램")
fig, ax = plt.subplots()
ax.hist(f_denoised.ravel(), bins=256, range=(0, 256), color="gray")
ax.axvline(th_val, color="red", linestyle="--", label=f"threshold = {th_val}")
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Frequency")
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------
# 5. Thresholding + Morphology (closing → opening)
# ---------------------------------------------------------------
st.header("4. 이진화 마스크")
f_mask = f_denoised > th_val
f_mask = nd.binary_closing(f_mask, iterations=iters)
f_mask = nd.binary_opening(f_mask, iterations=iters)
show_image(f_mask)

# ---------------------------------------------------------------
# 6. Connected component labeling
# ---------------------------------------------------------------
st.header("5. 연결 성분 라벨링")
labels, nums = nd.label(f_mask)
show_image(labels, cmap="Spectral_r")

if nums == 0:
    st.warning("검출된 성분이 없습니다. Threshold 값을 낮춰 보세요.")
    st.stop()

# ---------------------------------------------------------------
# 7. 각 성분의 픽셀 수
# ---------------------------------------------------------------
st.header("6. 성분별 픽셀 수")
cc = np.zeros((nums, 2))
lines = []
for i in range(1, nums + 1):
    cc[i - 1, 0] = i
    cc[i - 1, 1] = np.sum(labels == i)
    lines.append(f"index: {int(cc[i-1,0])}, number of pixels: {int(cc[i-1,1])}")
st.text("\n".join(lines))

ind_max = np.argmax(cc[:, 1])
ind_min = np.argmin(cc[:, 1])
st.write(f"**the largest component:** index {int(ind_max)}, # of pixels: {int(cc[ind_max,1])}")
st.write(f"**the smallest component:** index {int(ind_min)}, # of pixels: {int(cc[ind_min,1])}")

# ---------------------------------------------------------------
# 8. 최종 비교
# ---------------------------------------------------------------
st.header("7. 결과 비교")
fig = plt.figure(figsize=(16, 9))
plt.subplot(131)
plt.imshow(f, cmap="gray")
plt.axis("off")
plt.title("input image")
plt.subplot(132)
plt.imshow(labels == cc[ind_max, 0], cmap="Spectral_r")
plt.axis("off")
plt.title("the largest component")
plt.subplot(133)
plt.imshow(labels == cc[ind_min, 0], cmap="Spectral_r")
plt.axis("off")
plt.title("the smallest component")
st.pyplot(fig)
plt.close(fig)
