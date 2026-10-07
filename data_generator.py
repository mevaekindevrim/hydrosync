"""
data_generator.py
=================
HydroSync AI - Sentetik Veri Oluşturucu

7 günlük, 15 dakikalık aralıklı (toplam 672 nokta) zaman serisi üretir:
  - GPU_Utilization (%)        : gündüz yüksek / gece düşük + 2-3 adet "LLM Training Spike"
  - Ambient_WetBulb_Temp (C)   : günlük döngü, 18-32 °C
  - Water_Tariff_Price ($/m3)  : zaman-bazlı dinamik tarife (yoğun saatlerde pahalı)
  - Incoming_Water_TDS (ppm)   : şebeke suyu kalitesi, 300-400 ppm
  - LLM_Spike (bool)           : spike'ların etiketi (grafik ve doğrulama için)

Veri tamamen sentetiktir; aynı seed her zaman aynı veriyi üretir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Otokorelasyonlu (AR(1)) gürültü: ardışık noktalar birbirine yakın olur."""
    out = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _tariff_multiplier(hour: np.ndarray) -> np.ndarray:
    """Zaman-bazlı su tarifesi çarpanı (saat -> çarpan)."""
    mult = np.full_like(hour, 1.00, dtype=float)       # standart
    mult[(hour >= 0) & (hour < 6)] = 0.75               # gece / ucuz
    mult[(hour >= 10) & (hour < 14)] = 1.20             # gündüz orta yoğun
    mult[(hour >= 14) & (hour < 20)] = 1.60             # puant / pahalı
    return mult


def generate_dataset(
    days: int = 7,
    freq_min: int = 15,
    seed: int = 42,
    base_water_price: float = 2.5,
    n_spikes: int = 3,
    start: str = "2026-10-05 00:00",   # Pazartesi
) -> pd.DataFrame:
    """Sentetik veri setini üretir ve DataFrame olarak döner (varsayılan: 672 satır)."""
    rng = np.random.default_rng(seed)
    steps_per_day = int(24 * 60 / freq_min)
    n = days * steps_per_day

    ts = pd.date_range(start=start, periods=n, freq=f"{freq_min}min")
    hour = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    phase = 2.0 * np.pi * (hour - 9.0) / 24.0   # tepe noktası ~15:00, dip ~03:00

    # ---------------- GPU kullanımı (%) ----------------
    diurnal = 60.0 + 22.0 * np.sin(phase)                      # ~38 .. 82 %
    weekend = np.where(ts.dayofweek.to_numpy() >= 5, -6.0, 0.0)  # hafta sonu biraz düşük
    util = diurnal + weekend + _ar1(rng, n, phi=0.85, sigma=2.2)

    # LLM eğitim spike'ları: ani fırlama (tek adımda ~%95+), 4-6 saat sürer
    spike_flag = np.zeros(n, dtype=bool)
    centers = np.linspace(1.2, 5.6, n_spikes) * steps_per_day
    for c in centers:
        s = int(c + rng.integers(-8, 9))
        length = int(rng.integers(16, 25))          # 16-24 adım = 4-6 saat
        e = min(s + length, n)
        level = rng.uniform(95.0, 99.0)
        util[s:e] = np.maximum(util[s:e], level + rng.normal(0, 0.8, e - s))
        spike_flag[s:e] = True
        # spike sonrası hızlı ama kademeli düşüş (2 adım)
        for k in range(1, 3):
            if e + k - 1 < n:
                util[e + k - 1] = max(util[e + k - 1], level - 22.0 * k)
    util = np.clip(util, 5.0, 100.0)

    # ---------------- Yaş termometre sıcaklığı (°C) ----------------
    drift = np.convolve(_ar1(rng, n, 0.98, 0.15), np.ones(8) / 8, mode="same")  # günler arası yavaş değişim
    wet_bulb = 25.0 + 6.5 * np.sin(phase) + drift + rng.normal(0, 0.25, n)
    wet_bulb = np.clip(wet_bulb, 18.0, 32.0)

    # ---------------- Su tarifesi ($/m3) ----------------
    tariff = base_water_price * _tariff_multiplier(hour)
    tariff = tariff * (1.0 + rng.normal(0, 0.01, n))  # çok küçük piyasa gürültüsü

    # ---------------- Şebeke suyu TDS (ppm) ----------------
    slow = 28.0 * np.sin(2.0 * np.pi * np.arange(n) / (3.0 * steps_per_day) + 0.7)
    incoming_tds = np.clip(350.0 + slow + rng.normal(0, 6.0, n), 300.0, 400.0)

    return pd.DataFrame(
        {
            "Timestamp": ts,
            "GPU_Utilization": util.round(2),
            "Ambient_WetBulb_Temp": wet_bulb.round(2),
            "Water_Tariff_Price": tariff.round(4),
            "Incoming_Water_TDS": incoming_tds.round(1),
            "LLM_Spike": spike_flag,
        }
    )


def spike_windows(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """LLM_Spike bayrağından ardışık spike pencerelerini (başlangıç, bitiş) çıkarır."""
    flag = df["LLM_Spike"].to_numpy()
    windows, start = [], None
    for i, f in enumerate(flag):
        if f and start is None:
            start = i
        if (not f) and start is not None:
            windows.append((df["Timestamp"].iloc[start], df["Timestamp"].iloc[i - 1]))
            start = None
    if start is not None:
        windows.append((df["Timestamp"].iloc[start], df["Timestamp"].iloc[-1]))
    return windows


if __name__ == "__main__":
    data = generate_dataset()
    print(data.shape)
    print(data.describe().round(2).T)
    print("Spike pencereleri:", [(str(a), str(b)) for a, b in spike_windows(data)])
