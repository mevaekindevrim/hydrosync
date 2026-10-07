"""
optimizer.py
============
HydroSync AI - Reaktif (Baseline) vs Öngörücü (HydroSync AI) karşılaştırma algoritması

İki senaryo AYNI veri seti ve AYNI fizik motoru üzerinde koşturulur.

BASELINE (klasik, reaktif)
    * TDS > 1500 ppm olduğu an sabit debili blowdown vanası açılır (1450 ppm'de kapanır).
    * Vana debisi, tasarım yükünde CoC = 4 verecek şekilde seçilmiştir ve sabittir.
    * GPU yükü, LLM spike'ları, hava durumu ve tarifeler tamamen göz ardı edilir.

HYDROSYNC AI (öngörücü)
    a) LLM Spike Anticipation : GPU yükünü 30 dk önceden tahmin eder; spike gelmeden
       soğutma suyunu ön-soğutma ile kademeli hazırlar ve kütle dengesine (blowdown debisi,
       TDS payı) önceden müdahale eder.
    b) Tarife duyarlı blowdown : Pahalı puant saatlerde CoC'yi güvenli üst sınıra (7.5)
       çıkarıp blowdown'u ertelemek (TDS < 2800 ppm kaldığı sürece), ucuz/gece/serin
       saatlerde ise sistemi "flush" ederek blowdown'u o saatlere kaydırır.
    c) Dinamik PUE <-> WUE dengesi : Yaş termometre sıcaklığı ve su/elektrik maliyetine
       bakarak anlık "kuru destek" (dry-assist) kararı verir. Kuru destek suyu azaltır
       (WUE iyileşir) ama fan/pompa gücünü artırır (PUE kötüleşir). Karar maliyet bazlıdır.

ÖNEMLİ MODEL VARSAYIMLARI (jüriye sunarken şeffaf olun)
    * HydroSync'in 2800 ppm güvenli TDS sınırı, "Predictive Antiscalant Dosing & Water
      Conditioning" ile sağlandığı varsayılan bir tasarım kabulüdür; saha doğrulaması gerekir.
      Baseline standart kimyasal programla 1500 ppm eşiğinde çalışır.
    * Tahmin modeli (GPUForecaster) bir simülasyon yerine geçer: gerçek geleceğe, ayarlanabilir
      hata (gürültü) ekleyerek erişir. Gerçek sistemde Slurm/Kubernetes kuyruk verisi ile
      eğitilmiş bir model kullanılır.
    * Kuru destek (PUE/WUE), karbon ve elektrik parametreleri mühendislik yaklaşımlarıdır
      (OptimizerConfig içinde ayarlanabilir).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from physics_engine import CoolingTowerPhysics, PlantConfig


# ----------------------------------------------------------------------
# Konfigürasyon
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizerConfig:
    # ---- Baseline (reaktif) ----
    baseline_coc: float = 4.0                  # Vana, tasarım yükünde bu CoC'yi verecek şekilde boyutlanır
    baseline_tds_trigger_ppm: float = 1500.0   # Bu değerin üstünde vana açılır
    baseline_tds_release_ppm: float = 1450.0   # Bu değerin altında vana kapanır (histerezis)
    baseline_thermal_response: float = 0.5     # Reaktif soğutma kontrolcüsü tepkisi (0-1 / adım)

    # ---- HydroSync AI: CoC / TDS ----
    ai_target_coc: float = 6.0                 # Normal saatlerdeki hedef CoC (sidebar)
    ai_max_coc: float = 7.5                    # Pahalı saatlerde güvenli üst CoC
    ai_min_coc: float = 3.0
    ai_tds_limit_ppm: float = 2800.0           # Öngörücü şartlandırma ile güvenli TDS üst sınırı
    ai_safety_margin_ppm: float = 150.0        # Limitten bırakılan emniyet payı
    ai_flush_coc_drop: float = 1.5             # Ucuz saatlerde hedef CoC'den bu kadar aşağı flush

    # ---- HydroSync AI: tarife / hava ----
    expensive_ratio: float = 1.10              # fiyat >= ort * 1.10  -> pahalı (puant)
    cheap_ratio: float = 0.75                  # fiyat <= ort * 0.75  -> ucuz (gece)
    cool_wetbulb_quantile: float = 0.25        # En serin %25 yaş termometre saatleri -> flush'a uygun

    # ---- HydroSync AI: öngörü / spike ----
    forecast_horizon_steps: int = 2            # 2 x 15 dk = 30 dk
    forecast_noise_pct: float = 2.0            # Tahmin hatası (GPU % puanı, 1 sigma)
    spike_jump_threshold_pct: float = 15.0     # Öngörülen GPU sıçraması (% puan) -> spike alarmı
    precool_ramp: float = 0.5                  # Öngörülen ısı artışının ön-soğutmaya çevrilen oranı
    spike_headroom_ppm: float = 120.0          # Spike öncesi TDS hedefinden düşülen pay (ön-seyreltme)

    # ---- Blowdown vanası ----
    blowdown_capacity_factor: float = 3.0      # HydroSync vanası kapasitesi = 3 x baseline vana debisi
    valve_slew_frac: float = 0.5               # Adım başına maksimum debi değişimi (baseline debisinin oranı)

    # ---- Dinamik PUE <-> WUE (kuru destek) ----
    enable_dry_assist: bool = True             # Kuru destek (PUE<->WUE takası) açık / kapalı
    dry_assist_cap: float = 0.35               # Isı atımının en fazla bu kadarı kuru yoldan yapılabilir
    dry_twb_ref_c: float = 26.0                # Bu yaş termometre üstünde kuru destek etkisiz
    dry_twb_span_c: float = 8.0                # Kuru destek kapasitesi: (ref - Twb)/span ile artar
    dry_power_kappa: float = 0.02              # Kuru destek: ek güç = kappa * d * Q  (MW)
    electricity_price_usd_kwh: float = 0.08    # Elektrik fiyatı ($/kWh)

    # ---- Karbon ----
    water_embedded_kwh_per_m3: float = 1.0     # Suyun temini + atık su arıtımı için gömülü enerji (kWh/m3)
    grid_emission_kg_per_kwh: float = 0.40     # Şebeke emisyon faktörü (kgCO2/kWh)

    # ---- Genel ----
    seed: int = 7


# ----------------------------------------------------------------------
# GPU iş yükü tahmincisi (simülasyon)
# ----------------------------------------------------------------------
class GPUForecaster:
    """
    GPU kullanımını `horizon_steps` adım (varsayılan 30 dk) önceden tahmin eder.

    Bu sınıf bir ML modelinin yerine geçen simülasyondur: gerçek geleceği alır ve
    ayarlanabilir gürültü ekler. Üretimde Slurm/Kubernetes kuyruğundaki bekleyen/planlı
    işler + geçmiş telemetri ile eğitilmiş bir model bu sınıfın yerini alır.
    """

    def __init__(self, horizon_steps: int = 2, noise_pct: float = 2.0, seed: int = 7):
        self.horizon = int(horizon_steps)
        self.noise = float(noise_pct)
        self.rng = np.random.default_rng(seed)

    def predict(self, util_pct) -> np.ndarray:
        """pred[t] = t + horizon anındaki tahmini GPU kullanımı (%)."""
        u = np.asarray(util_pct, dtype=float)
        h = self.horizon
        future = np.concatenate([u[h:], np.repeat(u[-1], h)])   # son h adım için persistans
        noisy = future + self.rng.normal(0.0, self.noise, len(u))
        return np.clip(noisy, 0.0, 100.0)


# ----------------------------------------------------------------------
# Yardımcılar
# ----------------------------------------------------------------------
def _spike_starts(flag: np.ndarray) -> list[int]:
    """LLM_Spike bayrağındaki her ardışık bloğun başlangıç indeksi."""
    f = np.asarray(flag, dtype=bool)
    return [i for i in range(len(f)) if f[i] and (i == 0 or not f[i - 1])]


# ----------------------------------------------------------------------
# Baseline simülasyonu
# ----------------------------------------------------------------------
def run_baseline(df: pd.DataFrame, phys: CoolingTowerPhysics, cfg: OptimizerConfig) -> dict[str, np.ndarray]:
    """Reaktif sistem: TDS > 1500 ppm -> sabit debili blowdown. Başka hiçbir girdiye bakmaz."""
    n = len(df)
    dt = phys.cfg.dt_s
    cm = df["Incoming_Water_TDS"].to_numpy(float)
    twb = df["Ambient_WetBulb_Temp"].to_numpy(float)

    q = phys.heat_load_mw(phys.it_power_mw(df["GPU_Utilization"].to_numpy(float)))
    evap = phys.evaporation_rate(q, phys.wet_bulb_evap_factor(twb))

    b_fixed = float(phys.blowdown_rate(_design_evap(phys), cfg.baseline_coc))

    tds = np.zeros(n)
    blow = np.zeros(n)
    c = 0.5 * (cfg.baseline_tds_trigger_ppm + cfg.baseline_tds_release_ppm)  # kararlı bant ortası
    valve_open = False
    for t in range(n):
        if c > cfg.baseline_tds_trigger_ppm:
            valve_open = True
        elif c < cfg.baseline_tds_release_ppm:
            valve_open = False
        b = b_fixed if valve_open else 0.0
        m = evap[t] + b
        tds[t], blow[t] = c, b
        c = float(phys.tds_step(c, cm[t], m, b, evap[t]))

    # Reaktif kontrolcünün ısı yüküne gecikmeli yanıtı (1. derece gecikme)
    supply, deficit = q[0], np.zeros(n)
    for t in range(n):
        supply = supply + cfg.baseline_thermal_response * (q[t] - supply)
        deficit[t] = max(q[t] - supply, 0.0)

    return {"evap": evap, "blow": blow, "makeup": evap + blow, "tds": tds, "thermal_deficit_mw": deficit}


def _design_evap(phys: CoolingTowerPhysics) -> float:
    """Tasarım noktasında (GPU %100, katsayı 1.0) buharlaşma debisi (L/s)."""
    return float(phys.evaporation_rate(phys.heat_load_mw(phys.it_power_mw(100.0))))


# ----------------------------------------------------------------------
# HydroSync AI simülasyonu
# ----------------------------------------------------------------------
def run_hydrosync(df: pd.DataFrame, phys: CoolingTowerPhysics, cfg: OptimizerConfig) -> dict[str, np.ndarray]:
    """Öngörücü sistem: spike öngörüsü + tarife duyarlı blowdown + dinamik PUE/WUE kararı."""
    n = len(df)
    dt = phys.cfg.dt_s
    util = df["GPU_Utilization"].to_numpy(float)
    twb = df["Ambient_WetBulb_Temp"].to_numpy(float)
    price = df["Water_Tariff_Price"].to_numpy(float)
    cm = df["Incoming_Water_TDS"].to_numpy(float)

    # ---------------- a) 30 dk önceden GPU / ısı yükü tahmini ----------------
    forecaster = GPUForecaster(cfg.forecast_horizon_steps, cfg.forecast_noise_pct, cfg.seed)
    util_pred = forecaster.predict(util)
    q = phys.heat_load_mw(phys.it_power_mw(util))
    q_pred = phys.heat_load_mw(phys.it_power_mw(util_pred))
    p_it = phys.it_power_mw(util)

    # Spike alarmı: öngörülen sıçrama eşiği aşıyorsa spike gelmeden önce hazırlık başlar
    anticipation = (util_pred - util) > cfg.spike_jump_threshold_pct

    # Ön-soğutma: öngörülen ısı artışının bir kısmı şimdiden kademeli olarak karşılanır
    precool_mw = cfg.precool_ramp * np.maximum(q_pred - q, 0.0)
    q_eff = q + precool_mw

    f_evap = phys.wet_bulb_evap_factor(twb)
    e_full = phys.evaporation_rate(q_eff, f_evap)          # kuru destek öncesi buharlaşma
    e_pred_full = phys.evaporation_rate(q_pred, f_evap)    # 30 dk sonraki beklenen buharlaşma

    # ---------------- c) Dinamik PUE <-> WUE: kuru destek kararı ----------------
    # Kuru destek oranı d: ısı atımının d kadarı buharlaşmadan (kuru yoldan) yapılır.
    #   Su kazancı  : d * E * CoC/(CoC-1)   (hem buharlaşma hem blowdown azalır)
    #   Enerji bedeli: kappa * d * Q  (MW) ek fan/pompa gücü
    # Birim d başına maliyetler karşılaştırılır; su değeri > elektrik bedeli ise ve hava
    # yeterince serinse (Twb düşük) kuru destek açılır.
    d_cap_t = cfg.dry_assist_cap * np.clip((cfg.dry_twb_ref_c - twb) / cfg.dry_twb_span_c, 0.0, 1.0)
    coc_ref = max(cfg.ai_target_coc, 1.5)
    water_value = e_full * (coc_ref / (coc_ref - 1.0)) * price / 1000.0                  # $/s
    elec_cost = cfg.dry_power_kappa * q * 1000.0 * cfg.electricity_price_usd_kwh / 3600.0  # $/s
    dry_frac = np.where((water_value > elec_cost) & cfg.enable_dry_assist, d_cap_t, 0.0)
    extra_power_mw = cfg.dry_power_kappa * dry_frac * q

    evap = e_full * (1.0 - dry_frac)                       # gerçekleşen buharlaşma
    e_ctrl = np.maximum(evap, e_pred_full * (1.0 - dry_frac))  # kontrol: şimdiki ve 30 dk sonraki E'nin büyüğü

    # ---------------- b) Tarife / hava duyarlı mod belirleme ----------------
    mean_price = price.mean()
    expensive = price >= cfg.expensive_ratio * mean_price
    cheap = price <= cfg.cheap_ratio * mean_price
    cool = twb <= np.quantile(twb, cfg.cool_wetbulb_quantile)

    mode = np.where(expensive, "CONCENTRATE", np.where(cheap | cool, "FLUSH", "NORMAL"))
    coc_sp = np.where(
        mode == "CONCENTRATE",
        cfg.ai_max_coc,
        np.where(mode == "FLUSH", max(cfg.ai_target_coc - cfg.ai_flush_coc_drop, cfg.ai_min_coc), cfg.ai_target_coc),
    )

    # ---------------- Vana / TDS kontrol döngüsü ----------------
    b_fixed = float(CoolingTowerPhysics.blowdown_rate(_design_evap(phys), cfg.baseline_coc))
    b_max = cfg.blowdown_capacity_factor * b_fixed
    slew = cfg.valve_slew_frac * b_fixed
    hard_ceiling = cfg.ai_tds_limit_ppm - cfg.ai_safety_margin_ppm

    tds = np.zeros(n)
    blow = np.zeros(n)
    c = min(cfg.ai_target_coc * cm[0], hard_ceiling)           # başlangıç: hedef CoC'de kararlı rejim
    b_prev = float(CoolingTowerPhysics.blowdown_rate(evap[0], cfg.ai_target_coc))

    for t in range(n):
        setpoint = min(coc_sp[t] * cm[t], hard_ceiling)
        if anticipation[t]:
            setpoint -= cfg.spike_headroom_ppm                  # spike öncesi ön-seyreltme payı

        b_req = phys.required_blowdown_for_setpoint(c, cm[t], e_ctrl[t], setpoint)
        b_tgt = min(b_req, b_max)
        if c >= hard_ceiling:
            b = b_max                                           # acil durum: kireçlenme sınırına yaklaşıldı
        else:
            b = float(np.clip(b_tgt, b_prev - slew, b_prev + slew))  # vana kademeli açılıp kapanır

        m = evap[t] + b
        tds[t], blow[t] = c, b
        c = float(phys.tds_step(c, cm[t], m, b, evap[t]))
        b_prev = b

    return {
        "evap": evap, "blow": blow, "makeup": evap + blow, "tds": tds,
        "mode": mode, "dry_frac": dry_frac, "extra_power_mw": extra_power_mw,
        "precool_mw": precool_mw, "anticipation": anticipation,
        "util_pred": util_pred, "q": q, "q_pred": q_pred, "p_it": p_it,
        "expensive": expensive,
    }


# ----------------------------------------------------------------------
# Karşılaştırma + metrikler
# ----------------------------------------------------------------------
def run_comparison(
    df: pd.DataFrame,
    plant_cfg: PlantConfig | None = None,
    opt_cfg: OptimizerConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Baseline ve HydroSync AI'ı aynı veri üzerinde çalıştırır; (sonuç tablosu, metrikler) döner."""
    plant_cfg = plant_cfg or PlantConfig()
    opt_cfg = opt_cfg or OptimizerConfig()
    phys = CoolingTowerPhysics(plant_cfg)
    dt = plant_cfg.dt_s
    hours = dt / 3600.0

    bl = run_baseline(df, phys, opt_cfg)
    hs = run_hydrosync(df, phys, opt_cfg)

    price = df["Water_Tariff_Price"].to_numpy(float)
    cm = df["Incoming_Water_TDS"].to_numpy(float)

    # Interval başına su (L) ve maliyet ($)
    bl_water = bl["makeup"] * dt
    hs_water = hs["makeup"] * dt
    bl_cost = bl_water / 1000.0 * price
    hs_water_cost = hs_water / 1000.0 * price
    hs_elec_cost = hs["extra_power_mw"] * 1000.0 * hours * opt_cfg.electricity_price_usd_kwh
    hs_cost = hs_water_cost + hs_elec_cost

    res = pd.DataFrame(
        {
            "Timestamp": df["Timestamp"].to_numpy(),
            "GPU_Utilization": df["GPU_Utilization"].to_numpy(),
            "GPU_Pred_Util": hs["util_pred"],
            "LLM_Spike": df["LLM_Spike"].to_numpy(),
            "Ambient_WetBulb_Temp": df["Ambient_WetBulb_Temp"].to_numpy(),
            "Water_Tariff_Price": price,
            "Incoming_Water_TDS": cm,
            "P_IT_MW": hs["p_it"],
            "Heat_Load_MW": hs["q"],
            "Heat_Load_Pred_MW": hs["q_pred"],          # t anında yapılan, t+30dk için tahmin
            "BL_Evap_Lps": bl["evap"], "BL_Blowdown_Lps": bl["blow"], "BL_Makeup_Lps": bl["makeup"],
            "BL_TDS_ppm": bl["tds"], "BL_CoC": bl["tds"] / cm,
            "BL_Water_L": bl_water, "BL_Cost_USD": bl_cost,
            "HS_Evap_Lps": hs["evap"], "HS_Blowdown_Lps": hs["blow"], "HS_Makeup_Lps": hs["makeup"],
            "HS_TDS_ppm": hs["tds"], "HS_CoC": hs["tds"] / cm,
            "HS_Water_L": hs_water, "HS_Water_Cost_USD": hs_water_cost,
            "HS_Extra_Elec_Cost_USD": hs_elec_cost, "HS_Cost_USD": hs_cost,
            "HS_Mode": hs["mode"], "HS_Dry_Fraction": hs["dry_frac"],
            "HS_Extra_Power_MW": hs["extra_power_mw"], "HS_Precool_MW": hs["precool_mw"],
            "HS_Cooling_Staged_MW": hs["q"] + hs["precool_mw"],   # ön-soğutma dahil hazırlanan kapasite
            "HS_Anticipation": hs["anticipation"],
        }
    )

    # Karşı-olgusal koşu: kuru destek KAPALI iken HydroSync ne kadar su tasarrufu sağlar?
    # (Tasarrufun ne kadarının CoC/tarife optimizasyonundan, ne kadarının kuru destekten geldiğini gösterir.)
    hs_nodry = run_hydrosync(df, phys, replace(opt_cfg, enable_dry_assist=False))
    w_nodry = float((hs_nodry["makeup"] * dt).sum())

    # ---------------- Metrikler ----------------
    it_kwh = float((hs["p_it"] * 1000.0 * hours).sum())
    w_bl, w_hs = float(bl_water.sum()), float(hs_water.sum())
    wue_bl, wue_hs = w_bl / it_kwh, w_hs / it_kwh

    water_saved_l = w_bl - w_hs
    extra_kwh = float((hs["extra_power_mw"] * 1000.0 * hours).sum())
    carbon_avoided = water_saved_l / 1000.0 * opt_cfg.water_embedded_kwh_per_m3 * opt_cfg.grid_emission_kg_per_kwh
    carbon_added = extra_kwh * opt_cfg.grid_emission_kg_per_kwh

    pue_base = plant_cfg.pue_baseline
    pue_hs = (float((hs["p_it"] * pue_base).sum()) + float(hs["extra_power_mw"].sum())) / float(hs["p_it"].sum())

    starts = _spike_starts(df["LLM_Spike"].to_numpy())
    h = opt_cfg.forecast_horizon_steps
    anticipated = sum(bool(hs["anticipation"][max(s - h, 0):s].any()) for s in starts)

    expensive = hs["expensive"]
    days = len(df) * dt / 86400.0
    scale = 365.0 / days
    cost_saved = float(bl_cost.sum() - hs_cost.sum())

    metrics = {
        # Su
        "water_baseline_l": w_bl,
        "water_hydrosync_l": w_hs,
        "water_saved_l": water_saved_l,
        "water_saved_pct": 100.0 * water_saved_l / w_bl,
        # Maliyet
        "cost_baseline_usd": float(bl_cost.sum()),
        "cost_hydrosync_usd": float(hs_cost.sum()),
        "water_cost_saved_usd": float(bl_cost.sum() - hs_water_cost.sum()),
        "extra_electricity_cost_usd": float(hs_elec_cost.sum()),
        "cost_saved_usd": cost_saved,
        "cost_saved_pct": 100.0 * cost_saved / float(bl_cost.sum()),
        # WUE / PUE
        "wue_baseline_l_per_kwh": wue_bl,
        "wue_hydrosync_l_per_kwh": wue_hs,
        "wue_improvement_pct": 100.0 * (wue_bl - wue_hs) / wue_bl,
        "pue_baseline": pue_base,
        "pue_hydrosync": pue_hs,
        # CoC (ölçülen: C_tower / C_makeup zaman ortalaması)
        "avg_coc_baseline": float(res["BL_CoC"].mean()),
        "avg_coc_hydrosync": float(res["HS_CoC"].mean()),
        "max_tds_baseline_ppm": float(bl["tds"].max()),
        "max_tds_hydrosync_ppm": float(hs["tds"].max()),
        # Kireçlenme limitine kalan pay (%): her sistem KENDİ limitine göre (negatif = aşım)
        "scaling_headroom_baseline_pct": 100.0 * (1.0 - float(bl["tds"].max()) / plant_cfg.tds_critical_ppm),
        "scaling_headroom_hydrosync_pct": 100.0 * (1.0 - float(hs["tds"].max()) / opt_cfg.ai_tds_limit_ppm),
        # Kuru destek olmadan su tasarrufu (CoC + tarife + öngörü katkısı)
        "water_saved_pct_without_dry_assist": 100.0 * (w_bl - w_nodry) / w_bl,
        # Karbon (pozitif = fayda)
        "carbon_avoided_water_kg": carbon_avoided,
        "carbon_added_dry_assist_kg": carbon_added,
        "carbon_net_avoided_kg": carbon_avoided - carbon_added,
        "extra_electricity_kwh": extra_kwh,
        # Risk ve davranış
        "scaling_hours_baseline": float((bl["tds"] > plant_cfg.tds_critical_ppm).sum() * hours),
        "scaling_hours_hydrosync": float((hs["tds"] > opt_cfg.ai_tds_limit_ppm).sum() * hours),
        "baseline_thermal_lag_mwh_th": float(bl["thermal_deficit_mw"].sum() * hours),
        "spikes_total": len(starts),
        "spikes_anticipated": int(anticipated),
        "peak_makeup_share_baseline_pct": 100.0 * float(bl_water[expensive].sum()) / w_bl,
        "peak_makeup_share_hydrosync_pct": 100.0 * float(hs_water[expensive].sum()) / w_hs,
        # Yıllık ölçekleme (indikatif; 7 günlük simülasyondan)
        "annual_water_saved_m3": water_saved_l / 1000.0 * scale,
        "annual_cost_saved_usd": cost_saved * scale,
        "annual_carbon_net_avoided_t": (carbon_avoided - carbon_added) * scale / 1000.0,
        "sim_days": days,
    }
    return res, metrics


if __name__ == "__main__":
    from data_generator import generate_dataset

    data = generate_dataset()
    results, m = run_comparison(data, PlantConfig(capacity_mw=8.0), OptimizerConfig())
    for k, v in m.items():
        print(f"{k:38s} {v:,.3f}" if isinstance(v, float) else f"{k:38s} {v}")
