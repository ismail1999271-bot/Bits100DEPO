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
14. **Technology Scout Agent: yeni quant/AI/agent/data teknolojilerini düzenli araştırıp deney önerisi üretme**
15. **Institutional Intelligence: kurumsal araştırma, tahmin revizyonları, portföy riski, doküman/transkript analizi, sermaye hareketleri ve alternatif veri katmanları**

## Institutional Intelligence Layer

Kurumsal veri platformlarının faydalı yetenekleri tek bir vendor'a bağımlı olmadan normalize bir araştırma katmanında birleştirilir. Uygulama lisanslı veriyi taklit etmez; her sağlayıcı açıkça yapılandırılmış bir adapter üzerinden bağlanır.

`institutional_intelligence.py` şu kabiliyetleri sağlar:

- **Market layer:** fiyat/hacim, likidite, mikro yapı ve türev verileri için ortak skor sözleşmesi.
- **Fundamental layer:** finansallar, değerleme, peer-comps ve şirket kalitesi skorları için ortak alanlar.
- **Analyst layer:** EPS/gelir vb. tahminlerin point-in-time revizyon skorlaması.
- **Research layer:** haber, KAP/filing, earnings transcript, broker research ve uzman görüşü sentiment/kanıt agregasyonu.
- **Smart Money layer:** TEFAS/fon akışı ve tahmini maruziyet.
- **Institutional Consensus:** aynı yöneticiye ait çok sayıda fonu tek kurum olarak sayan ortaklaşma ölçümü.
- **Portfolio Risk:** top-1/top-5 ağırlık, Herfindahl endeksi ve effective positions.
- **Private-market/transactions:** PE/VC/M&A ve sermaye hareketleri için normalize veri sözleşmesi.
- **Alternative data:** web attention, sosyal, supply-chain ve diğer alternatif veri kümeleri için adapter alanı.
- **Evidence chain:** AI tarafından üretilen iddiaların kaynağı, gözlem zamanı ve güveni korunur.
- **Data coverage:** veri eksikliği gizlenmez; `institutional_data_coverage` ile raporlanır.
- **Disagreement flags:** farklı veri ailelerinin ters yönlü sinyallerini daha sonra risk katmanında cezalandırmaya uygun hale getirir.

Varsayılan composite ağırlıklar: **Market %50 + Smart Money %20 + Institutional Consensus %10 + Research %10 + Fundamentals %10**. Ağırlıklar `config/institutional_intelligence.yaml` üzerinden değiştirilebilir ve backtest ile doğrulanmadan üretim sinyali olarak kabul edilmez.

### Vendor capability mapping

- **Bloomberg Terminal:** gerçek zamanlı piyasa, haber, makro ve entegre araştırma iş akışları.
- **LSEG Workspace:** piyasa verisi, Reuters haberleri, analist tahminleri ve analitik.
- **FactSet:** portföy analitiği, benchmark, risk ve performans attribution.
- **S&P Capital IQ Pro:** şirket finansalları, değerleme, emsal şirket ve işlem verileri.
- **AlphaSense:** filing, transcript, araştırma ve haberlerde semantik/AI arama.
- **PitchBook:** private-company, PE/VC, M&A, yatırımcı ve sermaye hareketleri.
- **Preqin:** hedge fund, private equity, VC, private credit ve alternatif yatırım istihbaratı.

Bu isimler veri sağlayıcısı bağımlılığı anlamına gelmez; **yetenek haritasıdır**. Gerçek feed yalnızca yetkili/lisanslı kaynaklardan gelir.

## Institutional data fabric

`institutional_adapters.py` ile sekiz veri ailesi için ortak sözleşme tanımlanmıştır: `market`, `fundamentals`, `estimates`, `holdings`, `flows`, `research`, `transactions`, `alternative`.

Her kayıt için `observed_at` zorunludur; mevcutsa `as_of` korunur. `as_of > observed_at` olan kayıtlar look-ahead olarak reddedilir. AUM yoksa tahmini pozisyon değeri uydurulmaz. Portföy snapshot'ı trade ledger olarak yorumlanmaz.

`config/data_fabric.yaml` içindeki 10.000 kaynak kapasite hedefi ve günlük 48.000 simülasyon hedefi **mimari/araştırma kapasitesi hedefidir**; sistem hiçbir zaman var olmayan canlı feed'leri varmış gibi göstermemelidir.

## Technology Scout Agent

Projeye opsiyonel **Claude Agent SDK** tabanlı, read-only bir R&D ajanı eklendi. Anthropic'in Agent SDK'sı Claude Code yeteneklerini programatik ajanlara açıyor; Python SDK `query()` ve `ClaudeAgentOptions` üzerinden araçları ve çalışma alanını kontrol etmeye izin veriyor. urlClaude Agent SDK Pythonhttps://github.com/anthropics/claude-agent-sdk-python

Ajanın görevi:

- Two Sigma, Man AHL/Man Numeric, Jane Street, AQR gibi sistematik yatırım ekiplerinin kamuya açık araştırmalarını izlemek.
- QuantConnect/LEAN, akademik çalışmalar, alternatif veri, NLP, market microstructure ve agentic AI gelişmelerini taramak.
- Yeni fikri doğrudan sinyale çevirmek yerine **ADOPT / EXPERIMENT / WATCH / REJECT** kararı üretmek.
- Look-ahead leakage, survivorship bias, veri lisansı, likidite/manipülasyon, maliyet ve overfit risklerini kontrol etmek.
- Öneriyi mevcut KPI'lara bağlamak: Top-K precision, recall, lead time, net P&L, drawdown ve data quality.

Anthropic'in Agent Skills yaklaşımından da yararlanıyoruz: beceriler `SKILL.md` ile modüler ve yeniden kullanılabilir talimatlar olarak tanımlanabiliyor. Bizim BIST özel skill'imiz `.claude/skills/bist-technology-scout/SKILL.md` altında tutuluyor. urlAnthropic Agent Skillshttps://github.com/anthropics/skills

`config/research_watchlist.yaml` araştırma evrenini tanımlar. `.github/workflows/technology-scout.yml` ise API anahtarı sağlandığında haftalık araştırma çalıştırır; anahtar yoksa ana CI etkilenmez. Ajanın dosya değiştirme, shell veya işlem yapma yetkisi yoktur; üretim değişiklikleri deney → backtest → CI hattından geçmek zorundadır.

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

Bu katman günlük 06:00 raporunda şu formatta kullanılabilir: **Performans Liderleri → Yeni Yükselenler → Momentum İvmesi → Haber/KAP Katalizörü → Smart Money → Institutional Consensus → Tavan Skoru**.

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
- Institutional Intelligence katkısı: research/fundamental/estimate katmanlarının incremental lift'i
- Data coverage ve kaynak güvenilirliği

Sistem kaliteli fırsat yoksa zorla hisse seçmez ve `NO_QUALITY_SIGNAL` döndürebilir.

## Güvenlik

API anahtarları ve Telegram tokenları repoya commit edilmeyecek. `.env` yerel kullanım için, `.env.example` ise şablon olarak kullanılacak.
