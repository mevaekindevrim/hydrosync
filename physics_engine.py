"""
physics_engine.py
=================
HydroSync AI - Termodinamik ve Veri Merkezi Fizik Motoru

Soğutma kulesi (cooling tower) için kütle/enerji dengesi denklemlerini içerir.
Tüm fonksiyonlar numpy ile vektörize çalışır (skaler de verilebilir).

Birimler
--------
- Güç / ısı yükü      : MW
- Debi (E, B, M)      : L/s
- TDS                 : ppm (= mg/L)
- Hacim               : L
- Zaman adımı (dt)    : saniye

Temel denklemler
----------------
Q   = P_IT * (1 + (PUE - 1))                      -> kuleden atılacak ısı yükü
E   = 0.0018 * Q_ton                              -> buharlaşma (L/s)
CoC = M / B                                       -> devir daim sayısı
B   = E / (CoC - 1)                               -> blowdown (L/s)
M   = E + B                                       -> makeup (L/s)
C(t+1) = C(t) + dt * (M*C_makeup - B*C - E*0) / V -> TDS kütle dengesi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 1 soğutma tonu (refrigeration ton) = 3.5169 kW  ->  1 MW ≈ 284.35 ton
KW_PER_TON = 3.5169
TON_PER_MW = 1000.0 / KW_PER_TON


@dataclass(frozen=True)
class PlantConfig:
    """Veri merkezi + soğutma kulesi tesis parametreleri."""

    capacity_mw: float = 8.0               # Maksimum IT gücü (0-10 MW aralığı)
    pue_baseline: float = 1.35             # Baseline PUE (IT dışı yükler dahil toplam ısı)
    idle_power_frac: float = 0.25          # GPU boşta iken IT gücünün oranı
    basin_volume_m3_per_mw: float = 15.0   # Kule havuzu + devre su hacmi (m3 / MW)
    evap_coeff_lps_per_ton: float = 0.0018  # E = 0.0018 * Q_ton  (L/s)
    tds_critical_ppm: float = 1500.0       # Standart kimyasal programda kireçlenme eşiği
    dt_s: float = 900.0                    # 15 dakikalık zaman adımı


class CoolingTowerPhysics:
    """Soğutma kulesi fizik hesaplarını toplayan modüler sınıf."""

    def __init__(self, cfg: PlantConfig | None = None):
        self.cfg = cfg or PlantConfig()

    # ------------------------------------------------------------------
    # Tesis geometrisi
    # ------------------------------------------------------------------
    @property
    def volume_l(self) -> float:
        """Toplam sistem su hacmi V_tower (litre)."""
        return self.cfg.capacity_mw * self.cfg.basin_volume_m3_per_mw * 1000.0

    # ------------------------------------------------------------------
    # IT yükü ve ısı yükü
    # ------------------------------------------------------------------
    def it_power_mw(self, gpu_util_pct):
        """GPU kullanımına (%) bağlı dinamik IT gücü: 0 .. capacity_mw (MW)."""
        u = np.clip(np.asarray(gpu_util_pct, dtype=float), 0.0, 100.0) / 100.0
        idle = self.cfg.idle_power_frac
        return self.cfg.capacity_mw * (idle + (1.0 - idle) * u)

    def heat_load_mw(self, p_it_mw):
        """Q = P_IT * (1 + (PUE_baseline - 1))  (MW)."""
        return np.asarray(p_it_mw, dtype=float) * (1.0 + (self.cfg.pue_baseline - 1.0))

    @staticmethod
    def mw_to_tons(q_mw):
        """MW -> soğutma tonu."""
        return np.asarray(q_mw, dtype=float) * TON_PER_MW

    # ------------------------------------------------------------------
    # Su dengesi
    # ------------------------------------------------------------------
    @staticmethod
    def wet_bulb_evap_factor(wet_bulb_c):
        """
        Yaş termometre sıcaklığına bağlı buharlaşma düzeltmesi (mühendislik yaklaşımı).

        Hava ıslak-termometre sıcaklığı yükseldikçe duyulur ısı transferi azalır, ısı
        atımının daha büyük payı buharlaşmayla gerçekleşir. 25 °C'de katsayı 1.0'dır,
        her °C için ~%1 değişir ve [0.85, 1.15] aralığına sınırlandırılır.
        """
        f = 1.0 + 0.01 * (np.asarray(wet_bulb_c, dtype=float) - 25.0)
        return np.clip(f, 0.85, 1.15)

    def evaporation_rate(self, q_mw, evap_factor=1.0):
        """E = 0.0018 * Q_tons  (L/s). evap_factor: opsiyonel yaş-termometre düzeltmesi."""
        return self.cfg.evap_coeff_lps_per_ton * self.mw_to_tons(q_mw) * evap_factor

    @staticmethod
    def blowdown_rate(evaporation_lps, coc):
        """B = E / (CoC - 1)  (L/s)."""
        coc = np.maximum(np.asarray(coc, dtype=float), 1.0 + 1e-6)
        return np.asarray(evaporation_lps, dtype=float) / (coc - 1.0)

    @staticmethod
    def makeup_rate(evaporation_lps, blowdown_lps):
        """M = E + B  (L/s)."""
        return np.asarray(evaporation_lps, dtype=float) + np.asarray(blowdown_lps, dtype=float)

    @staticmethod
    def coc_from_flows(makeup_lps, blowdown_lps):
        """CoC = M / B. Blowdown sıfırsa CoC tanımsızdır (NaN döner)."""
        m = np.asarray(makeup_lps, dtype=float)
        b = np.asarray(blowdown_lps, dtype=float)
        return np.divide(m, b, out=np.full(np.broadcast(m, b).shape, np.nan), where=b > 1e-9)

    @staticmethod
    def steady_state_tds(c_makeup_ppm, coc):
        """Kararlı rejimde kule suyu TDS değeri: C = CoC * C_makeup."""
        return np.asarray(c_makeup_ppm, dtype=float) * np.asarray(coc, dtype=float)

    # ------------------------------------------------------------------
    # TDS kütle dengesi
    # ------------------------------------------------------------------
    def tds_step(self, c_tower, c_makeup, makeup_lps, blowdown_lps, evap_lps):
        """
        Bir zaman adımı ileri TDS dengesi:
            C(t+1) = C(t) + dt * (M*C_makeup - B*C_tower - E*0) / V_tower

        Buharlaşan su saftır (TDS = 0), bu nedenle E terimi 0 ile çarpılır.
        Sürüklenme (drift/windage) kayıpları ihmal edilmiştir.
        """
        net_mg_per_s = makeup_lps * c_makeup - blowdown_lps * c_tower - evap_lps * 0.0
        c_next = c_tower + self.cfg.dt_s * net_mg_per_s / self.volume_l
        return np.maximum(c_next, c_makeup)  # TDS makeup değerinin altına inemez

    def required_blowdown_for_setpoint(self, c_tower, c_makeup, evap_lps, c_setpoint):
        """
        Bir sonraki adımda TDS'yi `c_setpoint` değerine getirecek blowdown debisi (L/s).

        tds_step denkleminin B için tersi alınır:
            B = (E*C_makeup - (C_sp - C) * V/dt) / (C - C_makeup)
        Sonuç negatif ise 0 döner (blowdown gerekmez, TDS kendiliğinden yükselir).
        """
        denom = max(c_tower - c_makeup, 1e-6)
        b = (evap_lps * c_makeup - (c_setpoint - c_tower) * self.volume_l / self.cfg.dt_s) / denom
        return max(b, 0.0)

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------
    def scaling_risk(self, c_tower, limit_ppm: float | None = None):
        """TDS kritik eşiği aşıyorsa kireçlenme riski vardır (bool / bool dizi)."""
        limit = self.cfg.tds_critical_ppm if limit_ppm is None else limit_ppm
        return np.asarray(c_tower, dtype=float) > limit


if __name__ == "__main__":
    # Hızlı doğrulama: kararlı rejimde TDS = CoC * C_makeup olmalı
    phys = CoolingTowerPhysics(PlantConfig(capacity_mw=10.0))
    p_it = float(phys.it_power_mw(100))
    q = float(phys.heat_load_mw(p_it))
    e = float(phys.evaporation_rate(q))
    b = float(phys.blowdown_rate(e, 4.0))
    m = float(phys.makeup_rate(e, b))
    c = 1000.0
    for _ in range(2000):
        c = float(phys.tds_step(c, 350.0, m, b, e))
    print(f"P_IT={p_it:.1f} MW  Q={q:.2f} MW  E={e:.2f} L/s  B={b:.2f} L/s  M={m:.2f} L/s")
    print(f"CoC(M/B)={float(phys.coc_from_flows(m, b)):.2f}  TDS son durum={c:.1f} ppm (beklenen {4*350:.0f})")
