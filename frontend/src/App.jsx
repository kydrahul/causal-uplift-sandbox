import React, { useState, useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Activity, Clock, Star, Tags, Dices, Terminal, BarChart2, CheckCircle2, FlaskConical, Target, X, Plus, Info } from "lucide-react";

const InfoTooltip = ({ text }) => (
  <div className="relative group/tooltip inline-flex items-center ml-1.5 align-middle z-50">
    <Info className="w-3.5 h-3.5 text-zinc-500 hover:text-zinc-300 cursor-help transition-colors" />
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-2.5 bg-[#111113] border border-border/40 rounded-md text-[11px] leading-relaxed text-zinc-300 opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none shadow-xl text-center font-normal normal-case tracking-normal z-50">
      {text}
      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-[1px] border-4 border-transparent border-t-border/40"></div>
      <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-[2px] border-4 border-transparent border-t-[#111113]"></div>
    </div>
  </div>
);

function App() {
  const [features, setFeatures] = useState({
    activity_level: 5.0,
    recency: 15.0,
    avg_rating: 3.5,
    genre_entropy: 1.5,
    age_group: 25,
    is_male: 1,
    is_female: 0
  });

  const [predictions, setPredictions] = useState({ dml: null, s: null, value: null, segment: null, is_mismatch: false });
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([{ msg: "System initialized. Waiting for input...", type: "system", time: new Date() }]);
  const [serverStatus, setServerStatus] = useState("online");
  const [lastLatency, setLastLatency] = useState(0);
  const logContainerRef = useRef(null);

  const fullFeaturesRef = useRef({});

  const addLog = (msg, type = 'system') => {
    setLogs(prev => [...prev, { msg, type, time: new Date() }]);
  };

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const fetchRandomUser = async () => {
    setLoading(true);
    addLog("GET /api/random_user - Fetching random profile...", "request");
    const t0 = performance.now();
    try {
      const res = await fetch('/api/random_user');
      const data = await res.json();
      fullFeaturesRef.current = data.features;
      
      setFeatures(prev => ({
        ...prev,
        activity_level: data.features.activity_level ?? 5.0,
        recency: data.features.recency ?? 15.0,
        avg_rating: data.features.avg_rating ?? 3.5,
        genre_entropy: data.features.genre_entropy ?? 1.5,
        age_group: data.features.age_group ?? 25,
        is_male: data.features.is_male ?? 1,
        is_female: data.features.is_female ?? 0
      }));
      
      const lat = (performance.now() - t0).toFixed(0);
      setLastLatency(lat);
      setServerStatus("online");
      addLog(`200 OK - Fetched user index ${data.index}`, "response");
    } catch (e) {
      setServerStatus("offline");
      addLog(`Error fetching user: ${e.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  const runPrediction = async (currentFull = fullFeaturesRef.current) => {
    setLoading(true);
    
    const payload = {
      ...currentFull,
      activity_level: features.activity_level,
      recency: features.recency,
      avg_rating: features.avg_rating,
      genre_entropy: features.genre_entropy,
      age_group: features.age_group,
      is_male: features.is_male,
      is_female: features.is_female
    };
    
    addLog(`POST /api/predict ${JSON.stringify(payload).substring(0, 50)}...`, "request");
    const t0 = performance.now();
    
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: payload })
      });
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || `Server returned ${res.status}`);
      }
      
      const latency = (performance.now() - t0).toFixed(0);
      setLastLatency(latency);
      setServerStatus("online");
      
      setPredictions({
        dml: data.doubleml_uplift ?? null,
        s: data.s_learner_uplift ?? null,
        value: data.predicted_value ?? null,
        segment: data.segment ?? null,
        is_mismatch: data.is_mismatch ?? false
      });
      
      addLog(`200 OK (${latency}ms) -> DML: ${(data.doubleml_uplift*100).toFixed(2)}%, S: ${(data.s_learner_uplift*100).toFixed(2)}%`, "response");
    } catch (e) {
      setServerStatus("offline");
      addLog(`Error predicting: ${e.message}`, "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRandomUser();
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      runPrediction();
    }, 400);
    return () => clearTimeout(timer);
  }, [features]);

  const updateFeature = (key, val) => {
    setFeatures(prev => ({ ...prev, [key]: val }));
  };

  const formatPct = (val) => {
    if (val === null || val === undefined) return "--%";
    const pct = (val * 100).toFixed(2);
    return pct > 0 ? `+${pct}%` : `${pct}%`;
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-[#fafafa] p-4 md:p-8 font-sans">
      <div className="max-w-[1400px] mx-auto space-y-6">
        
        {/* Header */}
        <header className="mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Causal Inference Sandbox</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Uplift Modeling / Double Machine Learning
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Dynamic Latency and Status */}
            <div className="flex items-center gap-4 bg-[#111113] border border-border/40 px-4 py-2 rounded-md shadow-sm h-10 transition-colors duration-300">
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${serverStatus === 'online' ? 'bg-emerald-500' : 'bg-red-500 animate-pulse'}`}></span>
                <span className={`text-xs font-semibold uppercase tracking-wider transition-colors duration-300 ${serverStatus === 'online' ? 'text-zinc-300' : 'text-red-400'}`}>
                  {serverStatus === 'online' ? 'Online' : 'Offline'}
                </span>
              </div>
              <div className="w-px h-4 bg-border/40"></div>
              <div className="flex items-center gap-2">
                <Clock className={`w-3.5 h-3.5 ${serverStatus === 'online' ? 'text-muted-foreground' : 'text-red-400/50'}`} />
                <span className={`text-xs font-semibold uppercase tracking-wider transition-colors duration-300 ${serverStatus === 'online' ? 'text-zinc-300' : 'text-red-400'}`}>
                  {lastLatency > 0 ? `${lastLatency}ms` : '--ms'}
                </span>
              </div>
            </div>

            <a 
              href="https://github.com/kydrahul/causal-uplift-sandbox" 
              target="_blank" 
              rel="noreferrer"
              className="flex items-center gap-2 px-4 py-2 h-10 bg-[#18181b] hover:bg-zinc-800 border border-border/40 rounded-md text-sm font-medium transition-colors text-zinc-300 hover:text-white shadow-sm"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/></svg>
              <span>View on GitHub</span>
            </a>
          </div>
        </header>

        {predictions.is_mismatch && (
          <div className="mb-6 bg-red-500/10 border border-red-500/20 rounded-lg p-4 flex items-center justify-between shadow-lg backdrop-blur-md">
            <div className="flex items-center gap-4">
              <div className="bg-red-500/20 p-2 rounded-full">
                <Target className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <h3 className="text-red-400 font-semibold text-sm">Value Mismatch Warning</h3>
                <p className="text-red-400/80 text-xs mt-0.5">
                  High Causal Uplift, but Low Predicted Lifetime Value. Do not treat.
                </p>
              </div>
            </div>
            <Badge variant="outline" className="bg-red-500/10 text-red-400 border-red-500/20">
              Phantom Value
            </Badge>
          </div>
        )}

        {/* 3-Column Grid Layout */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch">
          
          {/* Column 1 (Leftmost): User Profile */}
          <Card className="bg-[#18181b] border-border/40 shadow-none flex flex-col h-[550px]">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <div className="space-y-1">
                <CardTitle className="text-base font-semibold">User Profile</CardTitle>
                <p className="text-[13px] text-muted-foreground">Tweak features to see real-time uplift.</p>
              </div>
              <Button variant="ghost" size="icon" onClick={fetchRandomUser} className="h-8 w-8 rounded-md hover:bg-white/10" title="Randomize User">
                <Dices className="h-4 w-4" />
              </Button>
            </CardHeader>
            <CardContent className="space-y-5 pt-4 flex-1 flex flex-col">
              
              {/* Activity Slider */}
              <div className="space-y-3">
                <div className="flex justify-between items-end">
                  <label className="text-[13px] font-medium leading-none text-zinc-200">Activity Level</label>
                  <span className="text-lg font-bold tracking-tight">{features.activity_level.toFixed(1)}</span>
                </div>
                <Slider max={10} step={0.1} value={[features.activity_level]} onValueChange={(v) => updateFeature('activity_level', v[0])} className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4" />
                <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
                  <span>0 (Min)</span>
                  <span>10 (Max)</span>
                </div>
              </div>

              {/* Recency Slider */}
              <div className="space-y-3">
                <div className="flex justify-between items-end">
                  <label className="text-[13px] font-medium leading-none text-zinc-200">Recency (Days)</label>
                  <span className="text-lg font-bold tracking-tight">{features.recency.toFixed(0)}</span>
                </div>
                <Slider max={60} step={1} value={[features.recency]} onValueChange={(v) => updateFeature('recency', v[0])} className="[&_[role=slider]]:h-4 [&_[role=slider]]:w-4" />
                <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
                  <span>0 (Recent)</span>
                  <span>60 (Dormant)</span>
                </div>
              </div>

              {/* Selects */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-[13px] font-medium text-zinc-200">Age Group</label>
                  <Select value={features.age_group.toString()} onValueChange={(v) => updateFeature('age_group', parseInt(v))}>
                    <SelectTrigger className="bg-[#09090b] border-border/40 h-9 text-[13px]"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-[#18181b] border-border/40">
                      <SelectItem value="18">18-24</SelectItem>
                      <SelectItem value="25">25-34</SelectItem>
                      <SelectItem value="35">35-44</SelectItem>
                      <SelectItem value="45">45-49</SelectItem>
                      <SelectItem value="50">50-55</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <label className="text-[13px] font-medium text-zinc-200">Gender</label>
                  <Select value={features.is_male ? "M" : "F"} onValueChange={(v) => { updateFeature('is_male', v==='M'?1:0); updateFeature('is_female', v==='F'?1:0); }}>
                    <SelectTrigger className="bg-[#09090b] border-border/40 h-9 text-[13px]"><SelectValue /></SelectTrigger>
                    <SelectContent className="bg-[#18181b] border-border/40">
                      <SelectItem value="M">Male</SelectItem>
                      <SelectItem value="F">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Hidden Features / Context */}
              <div className="mt-auto pt-4 border-t border-border/20">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-[13px] font-medium text-zinc-200 flex items-center">
                      Avg Rating
                      <InfoTooltip text="The average star rating (1-5) this user has given to movies. High rating indicates a satisfied user." />
                    </label>
                    <div className="flex items-center gap-2 bg-[#09090b] border border-border/40 rounded-md h-9 px-3 text-[13px] font-medium text-zinc-300">
                      <Star className="w-3.5 h-3.5 text-amber-500/70" />
                      {features.avg_rating.toFixed(1)}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-[13px] font-medium text-zinc-200 flex items-center">
                      Genre Entropy
                      <InfoTooltip text="A measure of taste diversity. High values mean the user explores many genres; low means they stick to one niche." />
                    </label>
                    <div className="flex items-center gap-2 bg-[#09090b] border border-border/40 rounded-md h-9 px-3 text-[13px] font-medium text-zinc-300">
                      <Tags className="w-3.5 h-3.5 text-emerald-500/70" />
                      {features.genre_entropy.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>

            </CardContent>
            <div className="p-4 pt-0">
               <Button className="w-full font-semibold text-[13px]" variant="default" onClick={fetchRandomUser}>
                 Load Next User
               </Button>
            </div>
          </Card>

          {/* Column 2 (Middle): DoubleML and S-Learner */}
          <div className="flex flex-col gap-6 h-[550px]">
            {/* DoubleML Card */}
            <Card className="bg-[#18181b] border-border/40 shadow-none flex-1 flex flex-col justify-between relative group">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-1">
                    <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">DoubleML Estimate</CardTitle>
                    <p className="text-[13px] text-zinc-300 flex items-center">
                      True Causal Uplift (CATE)
                      <InfoTooltip text="The true percentage change in probability that this user will return *because* they received a notification." />
                    </p>
                  </div>
                  <Badge variant="secondary" className="bg-white/10 text-white hover:bg-white/20 text-[10px] font-medium rounded-sm">SOTA</Badge>
                </div>
              </CardHeader>
              <CardContent className="pb-6 flex-1 flex flex-col">
                <div className="flex-1 flex flex-col justify-center">
                  <div key={predictions.dml} className="animate-in slide-in-from-bottom-2 fade-in duration-300">
                    <h2 className={`text-4xl lg:text-5xl font-bold tracking-tighter ${loading ? 'opacity-50' : 'opacity-100'} transition-opacity`}>
                      {formatPct(predictions.dml)}
                    </h2>
                  </div>
                </div>
                
                {/* Visual indicator of effect magnitude */}
                <div className="space-y-1.5 mt-4">
                  <div className="flex justify-between text-[11px] text-zinc-400 font-medium uppercase tracking-wider">
                    <span>Effect Magnitude</span>
                    <span className={predictions.dml > 0 ? "text-emerald-400" : "text-red-400"}>
                      {predictions.dml > 0.10 ? "Strong" : predictions.dml > 0 ? "Moderate" : "Negative"}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-zinc-800/50 rounded-full overflow-hidden border border-zinc-800">
                    <div 
                      className={`h-full transition-all duration-700 ease-out ${predictions.dml > 0 ? 'bg-emerald-500' : 'bg-red-500'}`} 
                      style={{ width: predictions.dml ? `${Math.min(100, Math.abs(predictions.dml) * 300)}%` : '0%' }}
                    ></div>
                  </div>
                </div>
              </CardContent>
              {/* Subtle background accent */}
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none transition-all group-hover:bg-blue-500/10"></div>
            </Card>

            {/* S-LEARNER Card */}
            <Card className="bg-[#18181b] border-border/40 shadow-none flex-1 flex flex-col justify-between relative group">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-1">
                    <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">S-Learner Baseline</CardTitle>
                    <p className="text-[13px] text-zinc-300 flex items-center">
                      Standard Naive ML
                      <InfoTooltip text="A simple predictive model that often overestimates the effect because it doesn't cleanly isolate the causal signal from random noise." />
                    </p>
                  </div>
                  <Badge variant="outline" className="bg-[#09090b] text-zinc-400 border-zinc-800 text-[10px]">Baseline</Badge>
                </div>
              </CardHeader>
              <CardContent className="pb-6 flex-1 flex flex-col">
                <div className="flex-1 flex flex-col justify-center">
                  <div key={predictions.s} className="animate-in slide-in-from-bottom-2 fade-in duration-300">
                    <h2 className={`text-4xl lg:text-5xl font-bold tracking-tighter ${loading ? 'opacity-50' : 'opacity-100'} transition-opacity`}>
                      {formatPct(predictions.s)}
                    </h2>
                  </div>
                </div>
                
                <div className="space-y-1.5 mt-4">
                  <div className="flex justify-between text-[11px] text-zinc-400 font-medium uppercase tracking-wider">
                    <span>Variance vs DoubleML</span>
                    <span className="text-amber-500">
                      {predictions.s !== null && predictions.dml !== null ? Math.abs((predictions.s - predictions.dml)*100).toFixed(1) + "%" : "--%"}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-zinc-800/50 rounded-full overflow-hidden border border-zinc-800">
                    <div 
                      className="h-full bg-amber-500 transition-all duration-700 ease-out"
                      style={{ width: predictions.s !== null && predictions.dml !== null ? `${Math.min(100, Math.abs(predictions.s - predictions.dml) * 500)}%` : '0%' }}
                    ></div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Column 3 (Rightmost): Lifetime Value and Minimized Log */}
          <div className="flex flex-col gap-6 h-[550px]">
            {/* Value Prediction Card */}
            <Card className="bg-[#18181b] border-border/40 shadow-none flex-1 flex flex-col justify-between relative group">
              <CardHeader className="pb-2">
                <div className="flex justify-between items-center">
                  <div className="space-y-1">
                    <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">Lifetime Value Model</CardTitle>
                    <p className="text-[13px] text-zinc-300 flex items-center">
                      E[Value | Retained]
                      <InfoTooltip text="The predicted value (e.g., watch time or revenue) this user will generate over their lifetime if they are successfully retained." />
                    </p>
                  </div>
                  <Badge variant="outline" className="bg-[#09090b] text-zinc-400 border-zinc-800 text-[10px]">LGBM</Badge>
                </div>
              </CardHeader>
              <CardContent className="pb-6 flex-1 flex flex-col">
                <div className="flex-1 flex flex-col justify-center">
                  <div key={predictions.value} className="animate-in slide-in-from-bottom-2 fade-in duration-300 flex items-baseline">
                    <h2 className={`text-4xl lg:text-5xl font-bold tracking-tighter ${loading ? 'opacity-50' : 'opacity-100'} transition-opacity`}>
                      {predictions.value !== null ? `${predictions.value.toFixed(1)}` : '--'}
                    </h2>
                    <span className="text-zinc-500 text-sm ml-2 font-semibold">pts</span>
                  </div>
                </div>
                
                <div className="space-y-1.5 mt-4">
                  <div className="flex justify-between text-[11px] text-zinc-400 font-medium uppercase tracking-wider">
                    <span>Target Segment</span>
                    <span className={predictions.is_mismatch ? 'text-red-400' : 'text-blue-400'}>
                      {predictions.segment ? predictions.segment : '--'}
                    </span>
                  </div>
                  <div className="h-2 w-full bg-zinc-800/50 rounded-full overflow-hidden border border-zinc-800">
                    <div 
                      className={`h-full transition-all duration-700 ease-out ${predictions.is_mismatch ? 'bg-red-500' : 'bg-blue-500'}`} 
                      style={{ width: predictions.segment ? (predictions.is_mismatch ? '30%' : '100%') : '0%' }}
                    ></div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Minimized Live Server Log */}
            <Card className="bg-[#18181b] border-border/40 shadow-none flex-1 flex flex-col overflow-hidden">
              <CardHeader className="flex flex-row items-center justify-between pb-2 shrink-0">
                <div className="space-y-1">
                  <CardTitle className="text-[13px] font-semibold">Live Server Log</CardTitle>
                </div>
                <Terminal className="w-4 h-4 text-muted-foreground" />
              </CardHeader>
              <CardContent className="flex-1 flex flex-col min-h-0 pb-4">
                <div ref={logContainerRef} className="flex-1 min-h-0 bg-black/40 rounded-lg p-3 font-mono text-[10px] overflow-y-auto custom-scrollbar border border-border/30">
                  {logs.map((l, i) => (
                    <div key={i} className={`mb-1 leading-relaxed
                      ${l.type === 'system' ? 'text-gray-500' : ''}
                      ${l.type === 'request' ? 'text-zinc-300' : ''}
                      ${l.type === 'response' ? 'text-emerald-400' : ''}
                      ${l.type === 'error' ? 'text-red-400' : ''}
                    `}>
                      <span className="opacity-50 select-none mr-1">[{l.time.toISOString().split('T')[1].slice(0,8)}]</span>
                      {l.msg}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

        </div>

        {/* Bottom Section: Project Details Tabs */}
        <Card className="bg-[#18181b] border-border/40 shadow-none mt-6">
          <CardHeader className="pb-4 border-b border-border/30">
            <CardTitle className="text-base font-semibold">Project Details</CardTitle>
            <p className="text-[13px] text-muted-foreground">Comprehensive overview of the uplift modeling pipeline.</p>
          </CardHeader>
          <CardContent className="p-0">
            <Tabs defaultValue="details" className="w-full">
              <TabsList className="w-full justify-start bg-transparent border-b border-border/20 rounded-none h-12 p-0 px-4 overflow-x-auto">
                <TabsTrigger value="goal" className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-white data-[state=active]:text-white data-[state=active]:shadow-none rounded-none text-muted-foreground text-[13px] h-full whitespace-nowrap">The Goal</TabsTrigger>
                <TabsTrigger value="how-it-works" className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-white data-[state=active]:text-white data-[state=active]:shadow-none rounded-none text-muted-foreground text-[13px] h-full whitespace-nowrap">How It Works</TabsTrigger>
                <TabsTrigger value="value-awareness" className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-white data-[state=active]:text-white data-[state=active]:shadow-none rounded-none text-muted-foreground text-[13px] h-full whitespace-nowrap">Business Impact</TabsTrigger>
                <TabsTrigger value="trust" className="data-[state=active]:bg-transparent data-[state=active]:border-b-2 data-[state=active]:border-white data-[state=active]:text-white data-[state=active]:shadow-none rounded-none text-muted-foreground text-[13px] h-full whitespace-nowrap">Trust & Performance</TabsTrigger>
              </TabsList>
              
              <div className="p-6">
                <TabsContent value="goal" className="space-y-6 mt-0 animate-in fade-in duration-300">
                  <div className="max-w-3xl space-y-4">
                    <h3 className="text-lg font-semibold text-zinc-100">Stop Wasting Notifications</h3>
                    <p className="text-[14px] text-zinc-400 leading-relaxed">
                      <strong>What is this?</strong> This is a dashboard that simulates an advanced AI system for retaining users. When you hit "Load Next User", it calculates whether sending a push notification will actually <em>cause</em> that user to come back to the platform.
                    </p>
                    <p className="text-[14px] text-zinc-400 leading-relaxed">
                      <strong>Why does this matter?</strong> Most companies blast marketing emails or notifications to everyone. This is incredibly wasteful. It annoys users who were going to come back anyway ("Sure Things"), and wastes money on users who will never return regardless ("Lost Causes"). 
                    </p>
                    <p className="text-[14px] text-zinc-400 leading-relaxed">
                      <strong>The Solution:</strong> We use Causal Machine Learning to find the <strong>Persuadables</strong>—the rare, highly-valuable users who will only return <em>because</em> they received your notification. By targeting only them, you maximize retention while minimizing annoyance and cost.
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="how-it-works" className="space-y-6 mt-0 animate-in fade-in duration-300">
                  <div className="max-w-3xl space-y-4">
                    <h3 className="text-lg font-semibold text-zinc-100">Correlation vs. Causation</h3>
                    <p className="text-[14px] text-zinc-400 leading-relaxed">
                      <strong>The Problem with Standard AI:</strong> Standard AI models (like the "S-Learner Baseline" above) are great at finding correlations, but terrible at finding causation. For example, standard AI might notice that users who buy premium items get more emails. It wrongly assumes "emails cause purchases." It gets confused by bias in the historical data.
                    </p>
                    <p className="text-[14px] text-zinc-400 leading-relaxed">
                      <strong>How we fix it:</strong> We use a state-of-the-art technique called <strong>Double Machine Learning (DoubleML)</strong>. Instead of just looking for patterns, DoubleML mathematically "scrubs" the bias out of the historical data before making a prediction. 
                    </p>
                    <p className="text-[14px] text-zinc-400 leading-relaxed">
                      <strong>The Result:</strong> It outputs the <em>True Causal Uplift (CATE)</em>. This is the exact percentage change in probability that a user will return solely because of the intervention. 
                    </p>
                  </div>
                </TabsContent>

                <TabsContent value="value-awareness" className="space-y-6 mt-0 animate-in fade-in duration-300">
                  <div className="max-w-5xl space-y-4">
                    <h3 className="text-lg font-semibold text-zinc-100">Are they worth saving?</h3>
                    <p className="text-[14px] text-zinc-400 leading-relaxed mb-6">
                      Predicting if we <em>can</em> save a user is only half the battle. We also need to know if they are <em>worth</em> saving. We run a second AI model (Lifetime Value Model) to predict how much value (watch time, revenue, etc.) a user will generate if they are successfully retained. By cross-referencing Causal Uplift with Lifetime Value, we segment users into four actionable quadrants:
                    </p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                      <div className="space-y-4">
                        <div className="bg-[#09090b]/50 p-4 rounded-md border border-border/30">
                          <h4 className="text-[13px] font-bold text-yellow-500 mb-1">Star Users (High Uplift, High Value)</h4>
                          <p className="text-[12px] text-zinc-400">The VIPs. They will leave if ignored, but will stay if notified. Because they spend a lot, this is where you should maximize your marketing budget.</p>
                        </div>
                        <div className="bg-[#09090b]/50 p-4 rounded-md border border-border/30">
                          <h4 className="text-[13px] font-bold text-blue-400 mb-1">Sure Things (Low Uplift, High Value)</h4>
                          <p className="text-[12px] text-zinc-400">They love the product and will stay regardless of what we do. Don't waste budget on them, just let them enjoy the app.</p>
                        </div>
                        <div className="bg-[#09090b]/50 p-4 rounded-md border border-border/30">
                          <h4 className="text-[13px] font-bold text-red-400 mb-1">Mismatch (High Uplift, Low Value)</h4>
                          <p className="text-[12px] text-zinc-400">"Phantom Value". They react positively to notifications, but their lifetime value is near zero. Wasting notifications on them creates fake engagement metrics.</p>
                        </div>
                        <div className="bg-[#09090b]/50 p-4 rounded-md border border-border/30">
                          <h4 className="text-[13px] font-bold text-zinc-500 mb-1">Lost Causes (Low Uplift, Low Value)</h4>
                          <p className="text-[12px] text-zinc-400">They don't engage and they don't spend. Ignore them completely.</p>
                        </div>
                      </div>
                      <div className="bg-[#111113] border border-border/30 rounded-xl p-2 flex justify-center shadow-inner h-full items-center">
                        <img src="/value_quadrant.png" alt="Value vs Uplift Quadrant Analysis" className="max-h-[400px] object-contain rounded-md opacity-90 hover:opacity-100 transition-opacity" />
                      </div>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="trust" className="space-y-6 mt-0 animate-in fade-in duration-300">
                  <div className="max-w-5xl space-y-4">
                    <h3 className="text-lg font-semibold text-zinc-100">Concrete Model Performance</h3>
                    <p className="text-[14px] text-zinc-400 leading-relaxed mb-6">
                      Instead of theoretical tests, we evaluate Uplift models using standard business metrics: the <strong>Qini Curve</strong> and <strong>AUUC (Area Under Uplift Curve)</strong>. These metrics prove how much actual business value the model captures compared to targeting users randomly.
                    </p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 items-stretch">
                      <div className="flex flex-col h-full">
                        <h4 className="text-[14px] font-semibold text-zinc-200 mb-2">Qini Curve (Cumulative Uplift)</h4>
                        <p className="text-[13px] text-zinc-400 mb-4">
                          The Qini curve shows how many "incremental" retentions we gain as we target more users. A steeper curve means the model successfully identifies the best targets first. Our DoubleML model captures significantly more uplift earlier than the baseline S-Learner.
                        </p>
                        <div className="bg-[#111113] border border-border/30 rounded-xl p-2 flex justify-center shadow-inner mt-auto">
                          <img src="/results/figures/qini_curves.png" alt="Qini Curve" className="w-full h-auto object-contain rounded-md opacity-90 hover:opacity-100 transition-opacity" />
                        </div>
                      </div>

                      <div className="flex flex-col h-full">
                        <h4 className="text-[14px] font-semibold text-zinc-200 mb-2">Area Under Uplift Curve (AUUC)</h4>
                        <p className="text-[13px] text-zinc-400 mb-4">
                          AUUC provides a concrete, numerical score for the models. Higher is better. The DoubleML model achieves a significantly higher AUUC than both random targeting and naive baseline models, proving its superiority in isolating the true causal signal.
                        </p>
                        <div className="bg-[#111113] border border-border/30 rounded-xl p-2 flex justify-center shadow-inner mt-auto">
                          <img src="/results/figures/metrics_bar_auuc.png" alt="AUUC Metrics" className="w-full h-auto object-contain rounded-md opacity-90 hover:opacity-100 transition-opacity" />
                        </div>
                      </div>
                    </div>
                  </div>
                </TabsContent>
              </div>

            </Tabs>
          </CardContent>
        </Card>

      </div>
    </div>
  );
}

export default App;
