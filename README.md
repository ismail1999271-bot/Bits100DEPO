# BIST Tavan Hunter

BIST'te günlük/haftalık güçlü yükseliş ve tavan adaylarını **tavan oluşmadan önce** tespit etmeye yönelik araştırma ve backtest platformu.

## Proje hedefi

Ana hedef, geçmiş BIST tavan olaylarının ortak özelliklerini veriyle keşfetmek ve bunlardan gerçekçi, işlem maliyetlerini dikkate alan bir sinyal motoru geliştirmektir.

> Bu proje yatırım tavsiyesi üretmez. Modeller araştırma/backtest amacıyla geliştirilir; canlı işlem öncesi paper trading zorunludur.

## İlk geliştirme sırası

1. Veri sözleşmeleri ve proje iskeleti
2. Tavan olaylarının çıkarılması (Tavan DNA)
3. Feature engineering
4. Pre-tavan sinyal motoru
5. Walk-forward backtest
6. Haber/KAP katalizör motoru
7. Market Regime / BIST Fear & Greed
8. Sosyal sentiment ve kaynak güvenilirliği
9. ML ranking
10. Telegram 06:00 raporu ve canlı pozisyon izleme
11. Performans leaderboard'u: 1 gün / 5 gün / aylık momentum ve sıralama değişimi
12. TEFAS/fon akışı → Smart Money katmanı
13. **Institutional Consensus: büyük portföy yönetim şirketleri arasında ortak hisse yoğunlaşması**

## Fon akışı / Smart Money katmanı

`fund_flow.py` ve `smart_money.py` ile kurumsal para akışı sinyali ayrı bir veri katmanı olarak modellenir:

- Fon bazında günlük giriş, çıkış ve net akış tutulur.
- Fonun kendi geçmişine göre olağandışı akışlar z-score ile işaretlenir; bugünkü gözlem eşik hesabına dahil edilmez.
- Fon akışı, son bilinen fon portföy ağırlıklarıyla eşleştirilerek **tahmini hisse maruziyeti** (`estimated_flow_try`) hesaplanır.
- Aynı yöntemle sektör bazında tahmini kurumsal akış çıkarılabilir.
- Günlük ranking motoru, mevcut piyasa skorunu varsayılan **%20 Smart Money ağırlığıyla** fon akışıyla harmanlayabilir.
- Güçlü pozitif/negatif tahmini akışlar sırasıyla `smart money accumulation` / `smart money distribution` gerekçesi üretir.
- Bu hesap gerçek işlem defteri değildir; fon girişinin portföy ağırlıklarıyla oransal dağıtıldığı bir exposure estimate'tir.

Veri kaynağı adapter'ları TEFAS/KAP/vendor verisini bu normalize modele dönüştürür. Canlı veri yokmuş gibi davranılmaz ve portföy tarihleri ile günlük akış tarihleri aynı değilse bu zaman farkı korunur.

## Institutional Consensus / Kurumsal Ortaklaşma

`institutional_consensus.py`, TEFAS'tan bağımsız ikinci bir Smart Money katmanıdır. Amaç, büyük portföy yönetim şirketlerinin aynı BIST hisselerinde ortaklaşmasını ölçmektir.

İzlenecek ana kurum evreni: Ak, İş, Yapı Kredi, Garanti, QNB, TEB, Deniz, Oyak, Ünlü, Tacirler, İnfo, Tera, Pusula, Ata, Azimut ve Albaraka Portföy.

Temel kurallar:

- Aynı şirketin birden fazla fonu **tek kurumsal yönetici** olarak sayılır; `manager_count` ile `fund_count` ayrı tutulur.
- KAP/portföy raporundaki ağırlıklar pozisyon snapshot'ıdır; günlük trade ledger değildir.
- AUM mevcutsa `fon AUM × portföy ağırlığı` ile tahmini pozisyon değeri hesaplanır; AUM yoksa veri uydurulmaz.
- `NEW`, `INCREASE`, `DECREASE`, `EXIT`, `STABLE` pozisyon değişimleri izlenir.
- Kurumsal konsensüs skoru 0–100 arasında; kurum genişliği, pozisyon ağırlığı, değişim, veri tazeliği ve yeni girişleri birleştirir.
- Veri tarihi `as_of` olarak korunur; ileri tarih/lookahead kullanılmaz.
- `consensus_map()` ranking motoruna doğrudan 0–100 skor sözlüğü üretir.

Bu katman ileride TEFAS flow + broker flow + fiyat/hacim + KAP katalizörü ile tek Smart Money Composite içinde birleştirilecektir.

## Performans leaderboard'u

Sosyal medyada görülen “ayın en çok kazandıran hisseleri” formatı projeye **tanımlayıcı bir veri katmanı** olarak eklenmiştir. `performance_board.py` ile:

- BIST evrenindeki hisseler trailing dönem getirisine göre sıralanır.
- 20 işlem günü, aylık performans için pratik varsayılan dönemdir.
- Önceki döneme göre getiri ivmesi (`acceleration_pct`) hesaplanır.
- Önceki leaderboard sırasına göre yükseliş/düşüş (`rank_change`) izlenir.
- Yeterli geçmişi olmayan hisseler otomatik dışarıda bırakılır.
- Bu tablo tek başına al/sat sinyali değildir; sinyal motoruna momentum bağlamı sağlamak için kullanılır.

Bu katman günlük 06:00 raporunda şu formatta kullanılabilir: **Performans Liderleri → Yeni Yükselenler → Momentum İvmesi → Haber/KAP Katalizörü → Smart Money → Tavan Skoru**.

## Temel KPI'lar

- Daily Hit-Day Rate: güçlü yükseliş/tavan olayının doğru yakalandığı gün oranı
- Top-K Precision (özellikle Top-5 / Top-20)
- Strong-upside / tavan Recall
- Sinyalin olay öncesi lead time'ı
- Net getiri, expectancy ve P&L
- Maksimum drawdown
- Komisyon, spread, slippage ve likidite sonrası performans
- Smart Money katkısı: pozitif/negatif fon akışı ayrışmasının sinyal precision ve lead time üzerindeki etkisi
- Institutional Consensus katkısı: kurum genişliği ve pozisyon artışlarının precision/lead time üzerindeki etkisi

Sistem kaliteli fırsat yoksa zorla hisse seçmez ve `NO_QUALITY_SIGNAL` döndürebilir.

## Güvenlik

API anahtarları ve Telegram tokenları repoya commit edilmeyecek. `.env` yerel kullanım için, `.env.example` ise şablon olarak kullanılacak.
