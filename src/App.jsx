import { useState, useMemo } from 'react';
import { AllCommunityModule, ModuleRegistry } from 'ag-grid-community'; 
import { AgGridReact } from 'ag-grid-react';
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-alpine.css";

// Register all Community features
ModuleRegistry.registerModules([AllCommunityModule]);

// Custom cell renderer for price changes with color coding
const PriceChangeRenderer = (params) => {
  const value = params.value;
  const isPositive = value >= 0;
  const color = isPositive ? '#10b981' : '#ef4444';
  const sign = isPositive ? '+' : '';
  
  return (
    <span style={{ color, fontWeight: 'bold' }}>
      {sign}{value.toFixed(2)}%
    </span>
  );
};

// Custom cell renderer for currency
const CurrencyRenderer = (params) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(params.value);
};

// Custom cell renderer for large numbers (volume)
const VolumeRenderer = (params) => {
  const value = params.value;
  if (value >= 1000000) {
    return `${(value / 1000000).toFixed(2)}M`;
  } else if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}K`;
  }
  return value.toLocaleString();
};

// Custom cell renderer for P&L with color coding
const PnLRenderer = (params) => {
  const value = params.value;
  const isPositive = value >= 0;
  const color = isPositive ? '#10b981' : '#ef4444';
  const sign = isPositive ? '+' : '';
  
  return (
    <span style={{ color, fontWeight: 'bold' }}>
      {sign}${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
    </span>
  );
};

function App() {
  // Generate synthetic market data
  const marketData = useMemo(() => {
    const symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'UBER'];
    const sectors = ['Technology', 'Technology', 'Technology', 'E-commerce', 'Automotive', 'Social Media', 'Semiconductors', 'Entertainment', 'Semiconductors', 'Semiconductors', 'Enterprise', 'SaaS', 'Software', 'Fintech', 'Transportation'];
    
    return symbols.map((symbol, index) => {
      const basePrice = 50 + Math.random() * 450;
      const changePercent = (Math.random() - 0.5) * 10;
      const volume = Math.floor(Math.random() * 50000000) + 1000000;
      
      return {
        symbol,
        company: symbol === 'AAPL' ? 'Apple Inc.' : 
                 symbol === 'MSFT' ? 'Microsoft Corp.' :
                 symbol === 'GOOGL' ? 'Alphabet Inc.' :
                 symbol === 'AMZN' ? 'Amazon.com Inc.' :
                 symbol === 'TSLA' ? 'Tesla Inc.' :
                 symbol === 'META' ? 'Meta Platforms Inc.' :
                 symbol === 'NVDA' ? 'NVIDIA Corp.' :
                 symbol === 'NFLX' ? 'Netflix Inc.' :
                 symbol === 'AMD' ? 'Advanced Micro Devices' :
                 symbol === 'INTC' ? 'Intel Corp.' :
                 symbol === 'ORCL' ? 'Oracle Corp.' :
                 symbol === 'CRM' ? 'Salesforce Inc.' :
                 symbol === 'ADBE' ? 'Adobe Inc.' :
                 symbol === 'PYPL' ? 'PayPal Holdings' :
                 'Uber Technologies',
        sector: sectors[index],
        price: basePrice,
        change: changePercent,
        volume: volume,
        marketCap: Math.floor(basePrice * volume * 0.1),
        high52w: basePrice * 1.3,
        low52w: basePrice * 0.7,
        pe: (10 + Math.random() * 40).toFixed(2),
        dividend: (Math.random() * 3).toFixed(2)
      };
    });
  }, []);

  // Generate synthetic portfolio data
  const portfolioData = useMemo(() => {
    const holdings = marketData.slice(0, 8).map((stock, index) => {
      const shares = Math.floor(Math.random() * 100) + 10;
      const avgPrice = stock.price * (0.85 + Math.random() * 0.3);
      const currentValue = shares * stock.price;
      const costBasis = shares * avgPrice;
      const pnl = currentValue - costBasis;
      const pnlPercent = ((pnl / costBasis) * 100);
      
      return {
        symbol: stock.symbol,
        company: stock.company,
        shares: shares,
        avgPrice: avgPrice,
        currentPrice: stock.price,
        costBasis: costBasis,
        currentValue: currentValue,
        pnl: pnl,
        pnlPercent: pnlPercent,
        allocation: (Math.random() * 20 + 5).toFixed(2)
      };
    });
    
    return holdings;
  }, [marketData]);

  // Generate synthetic orders data
  const ordersData = useMemo(() => {
    const orderTypes = ['BUY', 'SELL'];
    const statuses = ['FILLED', 'PARTIAL', 'PENDING'];
    
    return marketData.slice(0, 10).map((stock, index) => {
      const shares = Math.floor(Math.random() * 50) + 5;
      const orderPrice = stock.price * (0.98 + Math.random() * 0.04);
      const orderType = orderTypes[Math.floor(Math.random() * orderTypes.length)];
      const status = statuses[Math.floor(Math.random() * statuses.length)];
      const timestamp = new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000);
      
      return {
        id: `ORD-${1000 + index}`,
        symbol: stock.symbol,
        type: orderType,
        shares: shares,
        price: orderPrice,
        total: shares * orderPrice,
        status: status,
        timestamp: timestamp.toLocaleString(),
        orderDate: timestamp.toISOString().split('T')[0]
      };
    });
  }, [marketData]);

  // Market Watch Grid Column Definitions
  const marketColDefs = useMemo(() => [
    { 
      field: 'symbol', 
      headerName: 'Symbol',
      pinned: 'left',
      width: 100,
      filter: 'agTextColumnFilter',
      checkboxSelection: true,
      headerCheckboxSelection: true
    },
    { 
      field: 'company', 
      headerName: 'Company',
      width: 200,
      filter: 'agTextColumnFilter'
    },
    { 
      field: 'sector', 
      headerName: 'Sector',
      width: 150,
      filter: 'agSetColumnFilter',
      enableRowGroup: true
    },
    { 
      field: 'price', 
      headerName: 'Price',
      width: 120,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'change', 
      headerName: 'Change %',
      width: 120,
      filter: 'agNumberColumnFilter',
      cellRenderer: PriceChangeRenderer,
      cellStyle: { textAlign: 'right' },
      comparator: (valueA, valueB) => valueA - valueB
    },
    { 
      field: 'volume', 
      headerName: 'Volume',
      width: 130,
      filter: 'agNumberColumnFilter',
      cellRenderer: VolumeRenderer,
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'marketCap', 
      headerName: 'Market Cap',
      width: 150,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => {
        const value = params.value;
        if (value >= 1000000000) {
          return `$${(value / 1000000000).toFixed(2)}B`;
        }
        return `$${(value / 1000000).toFixed(2)}M`;
      },
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'pe', 
      headerName: 'P/E Ratio',
      width: 100,
      filter: 'agNumberColumnFilter',
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'dividend', 
      headerName: 'Dividend %',
      width: 120,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => `${params.value}%`,
      cellStyle: { textAlign: 'right' }
    }
  ], []);

  // Portfolio Grid Column Definitions
  const portfolioColDefs = useMemo(() => [
    { 
      field: 'symbol', 
      headerName: 'Symbol',
      pinned: 'left',
      width: 100,
      filter: 'agTextColumnFilter'
    },
    { 
      field: 'company', 
      headerName: 'Company',
      width: 200,
      filter: 'agTextColumnFilter'
    },
    { 
      field: 'shares', 
      headerName: 'Shares',
      width: 100,
      filter: 'agNumberColumnFilter',
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'avgPrice', 
      headerName: 'Avg Price',
      width: 120,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'currentPrice', 
      headerName: 'Current Price',
      width: 130,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'costBasis', 
      headerName: 'Cost Basis',
      width: 130,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' },
      type: 'numericColumn'
    },
    { 
      field: 'currentValue', 
      headerName: 'Current Value',
      width: 140,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' },
      type: 'numericColumn'
    },
    { 
      field: 'pnl', 
      headerName: 'P&L',
      width: 130,
      filter: 'agNumberColumnFilter',
      cellRenderer: PnLRenderer,
      cellStyle: { textAlign: 'right' },
      type: 'numericColumn'
    },
    { 
      field: 'pnlPercent', 
      headerName: 'P&L %',
      width: 100,
      filter: 'agNumberColumnFilter',
      cellRenderer: PriceChangeRenderer,
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'allocation', 
      headerName: 'Allocation %',
      width: 130,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => `${params.value}%`,
      cellStyle: { textAlign: 'right' },
      aggFunc: 'sum'
    }
  ], []);

  // Orders Grid Column Definitions
  const ordersColDefs = useMemo(() => [
    { 
      field: 'id', 
      headerName: 'Order ID',
      width: 120,
      filter: 'agTextColumnFilter'
    },
    { 
      field: 'symbol', 
      headerName: 'Symbol',
      width: 100,
      filter: 'agTextColumnFilter'
    },
    { 
      field: 'type', 
      headerName: 'Type',
      width: 80,
      filter: 'agSetColumnFilter',
      cellStyle: (params) => {
        if (params.value === 'BUY') {
          return { color: '#10b981', fontWeight: 'bold' };
        }
        return { color: '#ef4444', fontWeight: 'bold' };
      }
    },
    { 
      field: 'shares', 
      headerName: 'Shares',
      width: 100,
      filter: 'agNumberColumnFilter',
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'price', 
      headerName: 'Price',
      width: 120,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' }
    },
    { 
      field: 'total', 
      headerName: 'Total',
      width: 130,
      filter: 'agNumberColumnFilter',
      valueFormatter: (params) => CurrencyRenderer({ value: params.value }),
      cellStyle: { textAlign: 'right' },
      type: 'numericColumn'
    },
    { 
      field: 'status', 
      headerName: 'Status',
      width: 120,
      filter: 'agSetColumnFilter',
      cellStyle: (params) => {
        const statusColors = {
          'FILLED': '#10b981',
          'PARTIAL': '#f59e0b',
          'PENDING': '#6b7280'
        };
        return { color: statusColors[params.value] || '#000', fontWeight: 'bold' };
      }
    },
    { 
      field: 'timestamp', 
      headerName: 'Timestamp',
      width: 180,
      filter: 'agDateColumnFilter',
      sort: 'desc'
    }
  ], []);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    resizable: true,
    filter: true,
    floatingFilter: true
  }), []);

  // Calculate portfolio totals
  const portfolioTotals = useMemo(() => {
    const totalCostBasis = portfolioData.reduce((sum, holding) => sum + holding.costBasis, 0);
    const totalCurrentValue = portfolioData.reduce((sum, holding) => sum + holding.currentValue, 0);
    const totalPnL = totalCurrentValue - totalCostBasis;
    const totalPnLPercent = ((totalPnL / totalCostBasis) * 100);
    
    return {
      totalCostBasis,
      totalCurrentValue,
      totalPnL,
      totalPnLPercent
    };
  }, [portfolioData]);

  return (
    <div style={{ 
      padding: '0',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      minHeight: '100vh',
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    }}>
      {/* Header */}
      <div style={{
        background: 'linear-gradient(135deg, #1e3a8a 0%, #312e81 100%)',
        padding: '24px 32px',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        borderBottom: '1px solid rgba(255,255,255,0.1)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 style={{ 
              margin: '0 0 8px 0', 
              color: '#ffffff',
              fontSize: '32px',
              fontWeight: '700',
              letterSpacing: '-0.5px'
            }}>
              Trading Dashboard
            </h1>
            <p style={{ 
              margin: '0', 
              color: 'rgba(255,255,255,0.7)',
              fontSize: '14px',
              fontWeight: '400'
            }}>
              Real-time market data & portfolio analytics
            </p>
          </div>
          <div style={{ 
            backgroundColor: 'rgba(255,255,255,0.1)',
            backdropFilter: 'blur(10px)',
            padding: '12px 20px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>
            <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: '11px', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Live Market Status
            </div>
            <div style={{ color: '#10b981', fontSize: '16px', fontWeight: '700', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ 
                width: '8px', 
                height: '8px', 
                backgroundColor: '#10b981', 
                borderRadius: '50%',
                display: 'inline-block',
                animation: 'pulse 2s infinite'
              }}></span>
              MARKETS OPEN
            </div>
          </div>
        </div>
      </div>

      <div style={{ padding: '32px' }}>
        {/* Portfolio Summary Cards */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '20px',
          marginBottom: '32px'
        }}>
          <div style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            padding: '24px',
            borderRadius: '16px',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: '-20px',
              right: '-20px',
              width: '100px',
              height: '100px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '50%',
              filter: 'blur(20px)'
            }}></div>
            <div style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px', marginBottom: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Cost Basis
            </div>
            <div style={{ color: '#ffffff', fontSize: '28px', fontWeight: '700', position: 'relative', zIndex: 1 }}>
              {CurrencyRenderer({ value: portfolioTotals.totalCostBasis })}
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            padding: '24px',
            borderRadius: '16px',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: '-20px',
              right: '-20px',
              width: '100px',
              height: '100px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '50%',
              filter: 'blur(20px)'
            }}></div>
            <div style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px', marginBottom: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Current Value
            </div>
            <div style={{ color: '#ffffff', fontSize: '28px', fontWeight: '700', position: 'relative', zIndex: 1 }}>
              {CurrencyRenderer({ value: portfolioTotals.totalCurrentValue })}
            </div>
          </div>

          <div style={{
            background: portfolioTotals.totalPnL >= 0 
              ? 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)' 
              : 'linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)',
            padding: '24px',
            borderRadius: '16px',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: '-20px',
              right: '-20px',
              width: '100px',
              height: '100px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '50%',
              filter: 'blur(20px)'
            }}></div>
            <div style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px', marginBottom: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Total P&L
            </div>
            <div style={{ color: '#ffffff', fontSize: '28px', fontWeight: '700', position: 'relative', zIndex: 1 }}>
              {portfolioTotals.totalPnL >= 0 ? '+' : ''}{CurrencyRenderer({ value: portfolioTotals.totalPnL })}
            </div>
          </div>

          <div style={{
            background: portfolioTotals.totalPnLPercent >= 0
              ? 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
              : 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
            padding: '24px',
            borderRadius: '16px',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            border: '1px solid rgba(255,255,255,0.2)',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <div style={{
              position: 'absolute',
              top: '-20px',
              right: '-20px',
              width: '100px',
              height: '100px',
              background: 'rgba(255,255,255,0.1)',
              borderRadius: '50%',
              filter: 'blur(20px)'
            }}></div>
            <div style={{ color: 'rgba(255,255,255,0.9)', fontSize: '12px', marginBottom: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Return Rate
            </div>
            <div style={{ color: '#ffffff', fontSize: '28px', fontWeight: '700', position: 'relative', zIndex: 1 }}>
              {portfolioTotals.totalPnLPercent >= 0 ? '+' : ''}{portfolioTotals.totalPnLPercent.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Portfolio Grid */}
        <div style={{ marginBottom: '32px' }}>
          <div style={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            overflow: 'hidden',
            border: '1px solid rgba(255,255,255,0.3)'
          }}>
            <div style={{
              padding: '24px 28px',
              borderBottom: '1px solid #e5e7eb',
              background: 'linear-gradient(to right, #f9fafb, #ffffff)'
            }}>
              <h2 style={{ 
                margin: '0', 
                color: '#111827',
                fontSize: '20px',
                fontWeight: '700',
                letterSpacing: '-0.3px'
              }}>
                📊 Portfolio Holdings
              </h2>
              <p style={{
                margin: '4px 0 0 0',
                color: '#6b7280',
                fontSize: '13px'
              }}>
                Track your investments and performance
              </p>
            </div>
            <div 
              className="ag-theme-alpine" 
              style={{ 
                height: '400px', 
                width: '100%'
              }}
            >
              <AgGridReact
                rowData={portfolioData}
                columnDefs={portfolioColDefs}
                defaultColDef={defaultColDef}
                enableRangeSelection={true}
                rowSelection="multiple"
                suppressRowClickSelection={true}
                animateRows={true}
                groupDisplayType="groupRows"
              />
            </div>
          </div>
        </div>

        {/* Market Watch Grid */}
        <div style={{ marginBottom: '32px' }}>
          <div style={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            overflow: 'hidden',
            border: '1px solid rgba(255,255,255,0.3)'
          }}>
            <div style={{
              padding: '24px 28px',
              borderBottom: '1px solid #e5e7eb',
              background: 'linear-gradient(to right, #f9fafb, #ffffff)'
            }}>
              <h2 style={{ 
                margin: '0', 
                color: '#111827',
                fontSize: '20px',
                fontWeight: '700',
                letterSpacing: '-0.3px'
              }}>
                🌐 Market Watch
              </h2>
              <p style={{
                margin: '4px 0 0 0',
                color: '#6b7280',
                fontSize: '13px'
              }}>
                Real-time market data across sectors
              </p>
            </div>
            <div 
              className="ag-theme-alpine" 
              style={{ 
                height: '500px', 
                width: '100%'
              }}
            >
              <AgGridReact
                rowData={marketData}
                columnDefs={marketColDefs}
                defaultColDef={defaultColDef}
                enableRangeSelection={true}
                rowSelection="multiple"
                suppressRowClickSelection={true}
                animateRows={true}
                enableCharts={false}
                sideBar={false}
                rowGroupPanelShow="always"
              />
            </div>
          </div>
        </div>

        {/* Orders Grid */}
        <div>
          <div style={{
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)',
            borderRadius: '16px',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            overflow: 'hidden',
            border: '1px solid rgba(255,255,255,0.3)'
          }}>
            <div style={{
              padding: '24px 28px',
              borderBottom: '1px solid #e5e7eb',
              background: 'linear-gradient(to right, #f9fafb, #ffffff)'
            }}>
              <h2 style={{ 
                margin: '0', 
                color: '#111827',
                fontSize: '20px',
                fontWeight: '700',
                letterSpacing: '-0.3px'
              }}>
                📋 Recent Orders
              </h2>
              <p style={{
                margin: '4px 0 0 0',
                color: '#6b7280',
                fontSize: '13px'
              }}>
                Your trading activity and order history
              </p>
            </div>
            <div 
              className="ag-theme-alpine" 
              style={{ 
                height: '400px', 
                width: '100%'
              }}
            >
              <AgGridReact
                rowData={ordersData}
                columnDefs={ordersColDefs}
                defaultColDef={defaultColDef}
                enableRangeSelection={true}
                rowSelection="single"
                animateRows={true}
              />
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
