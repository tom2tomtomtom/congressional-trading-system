import { useState, useEffect } from 'react'
import { AlertTriangle, TrendingUp, Users, Calendar, DollarSign, Eye, Bell, Shield } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import './App.css'

// Mock data for demonstration
const mockAlerts = [
  {
    id: 1,
    member: "Nancy Pelosi",
    stock: "NVDA",
    amount: "$3,000,000",
    suspicionScore: 10.0,
    level: "EXTREME",
    date: "2025-06-21",
    factors: ["Very large trade", "Extreme legislative access", "Late filing", "Uses spouse account", "AI legislation timing"]
  },
  {
    id: 2,
    member: "Ron Wyden",
    stock: "NVDA", 
    amount: "$375,000",
    suspicionScore: 9.2,
    level: "EXTREME",
    date: "2025-06-16",
    factors: ["CHIPS Act timing", "Finance Committee Chair", "190 days advance knowledge"]
  },
  {
    id: 3,
    member: "Ro Khanna",
    stock: "AAPL",
    amount: "$32,500",
    suspicionScore: 8.0,
    level: "HIGH",
    date: "2025-06-23",
    factors: ["Tariff announcement timing", "Tech oversight committee", "7 days advance knowledge"]
  }
]

const mockTrades = [
  { date: "2025-06-16", member: "Ron Wyden", stock: "NVDA", amount: 375000, type: "Purchase", score: 9.2 },
  { date: "2025-06-21", member: "Nancy Pelosi", stock: "NVDA", amount: 3000000, type: "Purchase", score: 10.0 },
  { date: "2025-06-23", member: "Ro Khanna", stock: "AAPL", amount: 32500, type: "Sale", score: 8.0 },
  { date: "2025-06-24", member: "Josh Gottheimer", stock: "JPM", amount: 175000, type: "Purchase", score: 6.0 },
  { date: "2025-06-25", member: "Debbie Wasserman Schultz", stock: "TSLA", amount: 85000, type: "Purchase", score: 5.5 }
]

const mockLegislation = [
  { date: "2025-07-03", title: "AI Safety and Regulation Hearing", significance: 8, committees: "House Oversight" },
  { date: "2025-07-10", title: "CHIPS Act Extension Bill", significance: 9, committees: "Finance Committee" },
  { date: "2025-07-15", title: "Banking Regulation Updates", significance: 7, committees: "Banking Committee" }
]

const performanceData = [
  { member: "Ron Wyden", return: 123.8, market: 24.9 },
  { member: "Debbie Wasserman Schultz", return: 142.3, market: 24.9 },
  { member: "Nancy Pelosi", return: 65.0, market: 24.9 },
  { member: "Ro Khanna", return: 45.0, market: 24.9 },
  { member: "Josh Gottheimer", return: 38.2, market: 24.9 }
]

const sectorData = [
  { name: "Technology", value: 45, color: "#8884d8" },
  { name: "Financial", value: 25, color: "#82ca9d" },
  { name: "Energy", value: 15, color: "#ffc658" },
  { name: "Healthcare", value: 10, color: "#ff7300" },
  { name: "Other", value: 5, color: "#00ff00" }
]

function App() {
  const [activeTab, setActiveTab] = useState("dashboard")
  const [alertCount, setAlertCount] = useState(0)
  const [isMonitoring, setIsMonitoring] = useState(true)

  useEffect(() => {
    // Simulate real-time updates
    const interval = setInterval(() => {
      setAlertCount(prev => prev + Math.floor(Math.random() * 2))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const getAlertBadgeColor = (level) => {
    switch(level) {
      case "EXTREME": return "bg-red-600 text-white"
      case "HIGH": return "bg-orange-500 text-white"
      case "MEDIUM": return "bg-yellow-500 text-black"
      default: return "bg-gray-500 text-white"
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <Shield className="h-8 w-8 text-blue-600" />
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                  Congressional Insider Trading Monitor
                </h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Real-time detection of suspicious trading patterns
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`h-3 w-3 rounded-full ${isMonitoring ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
                <span className="text-sm font-medium">
                  {isMonitoring ? 'Monitoring Active' : 'Monitoring Paused'}
                </span>
              </div>
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setIsMonitoring(!isMonitoring)}
              >
                {isMonitoring ? 'Pause' : 'Resume'}
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="alerts">Alerts</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
            <TabsTrigger value="legislation">Legislation</TabsTrigger>
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
                  <AlertTriangle className="h-4 w-4 text-red-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-red-600">{mockAlerts.length}</div>
                  <p className="text-xs text-muted-foreground">
                    +{alertCount} since last hour
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Monitored Members</CardTitle>
                  <Users className="h-4 w-4 text-blue-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">535</div>
                  <p className="text-xs text-muted-foreground">
                    House + Senate members
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Suspicion Score</CardTitle>
                  <TrendingUp className="h-4 w-4 text-orange-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">7.8</div>
                  <p className="text-xs text-muted-foreground">
                    Above 7.0 threshold
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Financial Impact</CardTitle>
                  <DollarSign className="h-4 w-4 text-green-500" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">$3.4M</div>
                  <p className="text-xs text-muted-foreground">
                    Estimated insider profits
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Recent High-Risk Alerts */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Bell className="h-5 w-5 text-red-500" />
                  <span>High-Risk Alerts</span>
                </CardTitle>
                <CardDescription>
                  Most suspicious trading activities detected in real-time
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockAlerts.slice(0, 3).map((alert) => (
                    <Alert key={alert.id} className="border-red-200 bg-red-50 dark:bg-red-950">
                      <AlertTriangle className="h-4 w-4 text-red-600" />
                      <AlertTitle className="flex items-center justify-between">
                        <span>{alert.member} - {alert.stock}</span>
                        <Badge className={getAlertBadgeColor(alert.level)}>
                          {alert.level} ({alert.suspicionScore}/10)
                        </Badge>
                      </AlertTitle>
                      <AlertDescription>
                        <div className="mt-2 space-y-1">
                          <p><strong>Amount:</strong> {alert.amount} | <strong>Date:</strong> {alert.date}</p>
                          <p><strong>Risk Factors:</strong> {alert.factors.slice(0, 2).join(", ")}...</p>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Performance Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Congressional vs Market Performance</CardTitle>
                <CardDescription>
                  2024 returns comparison showing potential insider advantage
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="member" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="return" fill="#8884d8" name="Congressional Return %" />
                    <Bar dataKey="market" fill="#82ca9d" name="S&P 500 %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Alerts Tab */}
          <TabsContent value="alerts" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>All Suspicious Trading Alerts</CardTitle>
                <CardDescription>
                  Comprehensive list of detected insider trading patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockAlerts.map((alert) => (
                    <div key={alert.id} className="border rounded-lg p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <h3 className="font-semibold text-lg">{alert.member}</h3>
                          <Badge variant="outline">{alert.stock}</Badge>
                          <Badge className={getAlertBadgeColor(alert.level)}>
                            {alert.level}
                          </Badge>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-bold">{alert.suspicionScore}/10</div>
                          <div className="text-sm text-gray-500">{alert.date}</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Trade Amount</p>
                          <p className="text-xl font-bold text-green-600">{alert.amount}</p>
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Risk Factors</p>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {alert.factors.map((factor, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs">
                                {factor}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                      
                      <Progress value={alert.suspicionScore * 10} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Sector Distribution */}
              <Card>
                <CardHeader>
                  <CardTitle>Suspicious Trades by Sector</CardTitle>
                  <CardDescription>
                    Distribution of suspicious trading activity across market sectors
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={sectorData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {sectorData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Trading Timeline */}
              <Card>
                <CardHeader>
                  <CardTitle>Suspicion Score Timeline</CardTitle>
                  <CardDescription>
                    Tracking suspicious activity over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={mockTrades}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="score" stroke="#8884d8" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            {/* Top Suspicious Members */}
            <Card>
              <CardHeader>
                <CardTitle>Most Suspicious Members (Current)</CardTitle>
                <CardDescription>
                  Ranking based on cumulative suspicion scores and patterns
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {performanceData.map((member, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className="text-lg font-bold text-gray-500">#{idx + 1}</div>
                        <div>
                          <p className="font-semibold">{member.member}</p>
                          <p className="text-sm text-gray-500">2024 Return: {member.return}%</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-red-600">
                          +{(member.return - member.market).toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-500">vs Market</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Legislation Tab */}
          <TabsContent value="legislation" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Calendar className="h-5 w-5" />
                  <span>Upcoming Legislative Events</span>
                </CardTitle>
                <CardDescription>
                  Monitor legislative calendar for potential insider trading opportunities
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {mockLegislation.map((event, idx) => (
                    <div key={idx} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold">{event.title}</h3>
                        <Badge variant={event.significance >= 8 ? "destructive" : "secondary"}>
                          Impact: {event.significance}/10
                        </Badge>
                      </div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                        <p><strong>Date:</strong> {event.date}</p>
                        <p><strong>Committees:</strong> {event.committees}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Alert>
              <Eye className="h-4 w-4" />
              <AlertTitle>Monitoring Active</AlertTitle>
              <AlertDescription>
                System is actively monitoring for trades that correlate with upcoming legislative events. 
                High-significance events (8+/10) trigger enhanced monitoring for related sectors.
              </AlertDescription>
            </Alert>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default App

