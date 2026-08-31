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

## Temel KPI'lar

- Aylık 10+ yüksek kaliteli tavan fırsatını önceden yakalama kapasitesi
- Precision / recall
- Sinyalin tavan öncesi lead time'ı
- Net getiri ve expectancy
- Maksimum drawdown
- İşlem maliyeti ve slippage sonrası performans

## Güvenlik

API anahtarları ve Telegram tokenları repoya commit edilmeyecek. `.env` yerel kullanım için, `.env.example` ise şablon olarak kullanılacak.
