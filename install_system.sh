#!/usr/bin/env bash
# install_system.sh – Fedora 42 system packages for GTX 980 Ti + Stable Diffusion
# Run with: sudo ./install_system.sh
set -euo pipefail

echo "==> Checking that we are on Fedora..."
if ! grep -q "Fedora" /etc/os-release; then
  echo "This script is intended for Fedora. Aborting."
  exit 1
fi

echo "==> Enabling RPM Fusion (free + nonfree) if not already present..."
dnf install -y \
  https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
  https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm \
  || true   # already installed is fine

echo "==> Updating system..."
dnf update -y

echo "==> Installing NVIDIA 580xx legacy drivers (Maxwell / Pascal – GTX 900 / 10-series)..."
# 580xx is the RPM Fusion package series that still supports GTX 980 Ti
dnf install -y \
  xorg-x11-drv-nvidia-580xx \
  akmod-nvidia-580xx \
  xorg-x11-drv-nvidia-580xx-cuda

echo "==> Installing Python 3, venv, pip, build tools and common libs..."
dnf install -y \
  python3 \
  python3-pip \
  python3-devel \
  python3-venv \
  gcc \
  gcc-c++ \
  make \
  git \
  mesa-libGL \
  libglvnd-glx \
  libX11 \
  libXext

echo ""
echo "============================================================"
echo " System packages installed."
echo ""
echo " IMPORTANT: The NVIDIA kernel modules need to be built by"
echo " akmods. This can take 1–3 minutes. Then reboot:"
echo ""
echo "   sudo reboot"
echo ""
echo " After reboot verify with:"
echo "   nvidia-smi"
echo "============================================================"
