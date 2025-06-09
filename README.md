# 🤖 AI Trading Bot with Llama

Trading bot yang menggunakan Streamlit untuk UI dan mengintegrasikan Llama AI melalui SambaNova Cloud untuk memberikan sinyal trading berdasarkan analisis teknikal.

## ✨ Fitur Utama

- **UI Streamlit yang Interaktif**: Interface yang mudah digunakan untuk memilih trading pair dan melihat analisis
- **Integrasi Llama AI**: Menggunakan Llama-4-Maverick-17B-128E-Instruct untuk analisis pasar dan rekomendasi trading
- **Chart TradingView Embed**: Menampilkan chart real-time dari TradingView
- **Analisis Teknikal Lengkap**: RSI, MACD, Bollinger Bands, Moving Averages, Support/Resistance
- **Multi-Asset Support**: Forex, Crypto, Commodities, dan Stocks
- **Sinyal Trading Spesifik**: Entry, Take Profit, Stop Loss dengan level harga yang tepat
- **Docker Support**: Easy deployment dengan Docker dan docker-compose

## 🛠️ Instalasi

### Option 1: Docker (Recommended)

#### Quick Start
```bash
# Clone repository
git clone <repository-url>
cd tradingBot

# Setup environment variables
cp .env.example .env
# Edit .env file dan masukkan API key Anda

# Run with docker-compose
docker-compose up -d
```

Aplikasi akan berjalan di `http://localhost:8501`

#### Manual Docker Commands
```bash
# Build image
docker build -t trading-bot .

# Run container
docker run -d \
  --name trading-bot \
  -p 8501:8501 \
  -e SAMBANOVA_API_KEY=your_api_key_here \
  trading-bot
```

### Option 2: Local Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd tradingBot
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Setup Environment Variables
```bash
cp .env.example .env
```

Edit file `.env` dan tambahkan API key SambaNova Anda:
```bash
SAMBANOVA_API_KEY=your_actual_api_key_here
```

#### 4. Dapatkan SambaNova API Key
1. Kunjungi [SambaNova Cloud](https://cloud.sambanova.ai/)
2. Daftar atau login ke akun Anda
3. Buat API key baru
4. Copy API key ke file `.env`

## 🚀 Cara Menjalankan

### Docker (Production Ready)
```bash
# Development mode (with logs)
docker-compose up

# Production mode (background)
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f trading-bot
```

### Local Development
```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## 📦 Docker Configuration

### Environment Variables
File `.env` untuk konfigurasi:
```bash
SAMBANOVA_API_KEY=your_sambanova_api_key_here
```

### Docker Features
- **Multi-stage build** untuk optimasi size
- **Non-root user** untuk security
- **Health checks** untuk monitoring
- **Auto restart** policy
- **Volume mounting** untuk persistent configuration

### Production Deployment
```bash
# Build production image
docker build -t trading-bot:production .

# Deploy dengan custom network
docker network create trading-network
docker run -d \
  --name trading-bot-prod \
  --network trading-network \
  -p 8501:8501 \
  --restart unless-stopped \
  --env-file .env \
  trading-bot:production
```

## 📊 Cara Menggunakan

### 1. Konfigurasi API Key
- Masukkan SambaNova API key di sidebar
- Pilih kategori asset (Forex, Crypto, Commodities, Stocks)
- Pilih trading pair yang ingin dianalisis

### 2. Chart Settings
- Pilih period data (1d, 5d, 1mo)
- Pilih interval chart (1m, 2m, 5m, 15m, 30m, 1h, 4h, 1d)
- Enable auto-refresh untuk data real-time

### 3. Analisis Chart
- Chart akan menampilkan candlestick dengan indikator teknikal
- RSI, MACD, dan Bollinger Bands ditampilkan secara otomatis
- Support dan resistance levels diidentifikasi oleh sistem

### 4. Mendapatkan Sinyal Trading
- Klik tombol "🔍 Get Trading Signal"
- AI akan menganalisis kondisi pasar saat ini
- Raw data yang dikirim ke AI akan ditampilkan di console/terminal
- Rekomendasi akan ditampilkan dengan format:
  - **TRADING SIGNAL**: LONG/SHORT/WAIT
  - **ENTRY PRICE**: Level harga masuk yang tepat
  - **TAKE PROFIT**: Target profit
  - **STOP LOSS**: Level stop loss
  - **CONFIDENCE**: Tingkat kepercayaan analisis
  - **REASONING**: Penjelasan analisis

### 5. TradingView Integration
- Chart TradingView embed tersedia di sidebar kanan
- Dapat digunakan untuk analisis tambahan
- Symbol otomatis disesuaikan untuk TradingView format

## 🔧 Supported Assets

### Forex Major Pairs
- EURUSD, GBPUSD, USDJPY, USDCHF
- AUDUSD, USDCAD, NZDUSD

### Commodities  
- **Gold (XAU/USD)**: GC=F
- **Silver (XAG/USD)**: SI=F
- **Crude Oil (WTI)**: CL=F
- **Natural Gas**: NG=F

### Cryptocurrencies
- BTC-USD, ETH-USD, ADA-USD
- DOT-USD, LINK-USD, UNI-USD

### Stocks
- AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA

## 🤖 AI Model Information

Bot menggunakan **Llama-4-Maverick-17B-128E-Instruct** dengan karakteristik:
- **17 Billion parameters** untuk analisis cepat dan akurat
- **128E context window** untuk memproses data kompleks
- **Optimized untuk trading analysis** dengan pattern recognition yang baik
- **Fast inference** cocok untuk real-time trading signals

## 🔧 Indikator Teknikal yang Digunakan

### 1. Moving Averages
- **SMA 20**: Simple Moving Average 20 periode
- **SMA 50**: Simple Moving Average 50 periode
- **EMA 9**: Exponential Moving Average 9 periode (fast EMA)
- **EMA 21**: Exponential Moving Average 21 periode (medium EMA)

### 2. Momentum Oscillators
#### RSI (Relative Strength Index)
- Period: 14
- Overbought: > 70
- Oversold: < 30

#### Stochastic Oscillator (%K, %D)
- Period: 14
- %K Line: Raw stochastic value
- %D Line: 3-period SMA of %K
- Overbought: > 80
- Oversold: < 20

#### Williams %R
- Period: 14
- Overbought: > -20
- Oversold: < -80
- Range: -100 to 0

#### CCI (Commodity Channel Index)
- Period: 20
- Overbought: > +100
- Oversold: < -100
- Normal range: -100 to +100

### 3. Trend Indicators
#### MACD (Moving Average Convergence Divergence)
- Fast EMA: 12
- Slow EMA: 26
- Signal Line: 9
- Histogram: MACD - Signal Line

#### ADX (Average Directional Index)
- Period: 14
- Strong trend: > 25
- Weak trend: < 20
- Plus DI and Minus DI for direction

### 4. Volatility Indicators
#### Bollinger Bands
- Period: 20
- Standard Deviation: 2
- BB Position: Price position within bands (0-100%)
- BB Width: Band expansion measurement

#### ATR (Average True Range)
- Period: 14
- Measures market volatility
- Used for dynamic stop loss calculation

### 5. Volume Indicators
#### Volume Analysis
- Volume SMA: 20-period average
- Volume Ratio: Current vs average volume
- High volume: Ratio > 1.5

#### OBV (On Balance Volume)
- Cumulative volume based on price direction
- Confirms price movements with volume

### 6. Support & Resistance
#### Dynamic Levels
- Identifies recent highs and lows
- Top 3 resistance levels above current price
- Top 3 support levels below current price

#### Fibonacci Retracement
- 23.6%, 38.2%, 50.0%, 61.8% levels
- Based on 50-period high/low range
- Dynamic adjustment to market structure

## 💡 Contoh Sinyal Trading

```
TRADING SIGNAL: LONG
ENTRY PRICE: 2651.80
TAKE PROFIT: 2658.50
STOP LOSS: 2647.20
CONFIDENCE: High
REASONING: Gold showing bullish momentum with RSI oversold recovery and MACD bullish crossover. Price broke above key resistance at 2650 with strong volume confirmation.
```

## ⚠️ Risk Management

**PENTING**: Ini adalah tool analisis berbasis AI dan bukan saran investasi finansial.

### Prinsip Risk Management:
- Jangan pernah risk lebih dari 1-2% dari total modal per trade
- Selalu gunakan stop loss
- Lakukan analisis sendiri sebelum mengambil keputusan
- Pertimbangkan kondisi pasar dan berita fundamental
- Gunakan position sizing yang tepat

## 🐛 Troubleshooting

### Docker Issues
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs trading-bot

# Restart containers
docker-compose restart

# Rebuild image
docker-compose build --no-cache
```

### Common Errors
#### Error: "LLM not initialized"
- Pastikan API key SambaNova sudah benar
- Check file .env sudah ter-mount dengan benar
- Verifikasi bahwa akun SambaNova aktif

#### Error: "Unable to fetch market data"
- Pastikan symbol trading pair sudah benar
- Cek format symbol (misal: BTC-USD untuk crypto, EURUSD=X untuk forex)
- Verifikasi koneksi internet

#### Port already in use
```bash
# Check what's using port 8501
lsof -i :8501

# Use different port
docker-compose up -d --build -p 8502:8501
```

## 📝 Development

### Local Development dengan Docker
```bash
# Development dengan volume mounting
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Hot reload untuk development
docker run -d \
  -p 8501:8501 \
  -v $(pwd):/app \
  --env-file .env \
  trading-bot
```

### Debug Mode
Raw data yang dikirim ke AI akan ditampilkan di terminal:
```bash
# Run dengan logging
docker-compose logs -f trading-bot
```

## 🤝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Support

Jika Anda mengalami masalah atau memiliki pertanyaan:
1. Buka issue di GitHub repository
2. Sertakan error message lengkap
3. Jelaskan langkah-langkah yang sudah dicoba
4. Sertakan output dari `docker-compose logs` jika menggunakan Docker

## 🔮 Roadmap

- [ ] Tambahan indikator teknikal (Stochastic, Williams %R)
- [ ] Multiple model selection (GPT, Claude integration)
- [ ] Backtesting functionality
- [ ] Email/Telegram notifications
- [ ] Multiple timeframe analysis
- [ ] Portfolio management features
- [ ] Kubernetes deployment configs