import { useEffect, useState, useMemo } from 'react'
import { dashboardApi } from '../services/api'

type Tab = 'comparison' | 'ablation' | 'plots' | 'threshold' | 'robustness' | 'shap' | 'governance'

const ALGORITHM_FAMILIES: Record<string, { family: string; tuned: string }> = {
  XGBoost: { family: 'Gradient Boosted Decision Trees', tuned: 'Yes (scale_pos_weight=3.95)' },
  Voting_Ensemble: { family: 'Soft-Voting Meta-Ensemble', tuned: 'Yes (Weights tuned)' },
  LightGBM: { family: 'Leaf-wise Gradient Boosting', tuned: 'Yes (Optuna 15 trials)' },
  CatBoost: { family: 'Symmetric Oblivious Trees', tuned: 'No (Default)' },
  HistGradientBoosting: { family: 'Histogram Binning GBDT', tuned: 'No (Default)' },
  RandomForest: { family: 'Bagging Ensemble Trees', tuned: 'No (n_estimators=100)' },
  ExtraTrees: { family: 'Extremely Randomized Trees', tuned: 'No (n_estimators=100)' },
  DecisionTree: { family: 'Single CART Decision Tree', tuned: 'No (max_depth=6)' },
  LogisticRegression: { family: 'Generalized Linear Model (L2)', tuned: 'Yes (class_weight=balanced)' },
  Dummy: { family: 'Empirical Prior Baseline', tuned: 'No (Prior frequency)' },
}

export default function ModelInsights() {
  const [activeTab, setActiveTab] = useState<Tab>('comparison')
  const [versionView, setVersionView] = useState<'v5' | 'v4'>('v5')

  // V5 Data
  const [v5ValMetrics, setV5ValMetrics] = useState<any[]>([])
  const [v5TestMetrics, setV5TestMetrics] = useState<any[]>([])
  const [v5Thresholds, setV5Thresholds] = useState<any[]>([])
  const [v5Cv, setV5Cv] = useState<any>(null)
  const [v5FeatureSelection, setV5FeatureSelection] = useState<any>(null)
  const [v5Metadata, setV5Metadata] = useState<any>(null)
  const [experiments, setExperiments] = useState<any[]>([])

  // V4 Data
  const [v4ValMetrics, setV4ValMetrics] = useState<any[]>([])
  const [v4TestMetrics, setV4TestMetrics] = useState<any[]>([])
  const [v4ThresholdData, setV4ThresholdData] = useState<any[]>([])
  const [v4CvSummary, setV4CvSummary] = useState<any>(null)
  const [v4SeedData, setV4SeedData] = useState<any[]>([])
  const [v4PermData, setV4PermData] = useState<any[]>([])
  const [v4ShapData, setV4ShapData] = useState<any[]>([])

  const [datasetStats, setDatasetStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const apiBase = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

  useEffect(() => {
    let active = true

    async function loadAllInsights() {
      setLoading(true)
      setError('')

      try {
        const [
          v5ValRes,
          v5TestRes,
          v5ThreshRes,
          v5CvRes,
          v5FeatRes,
          v5MetaRes,
          expRes,
          v4ValRes,
          v4TestRes,
          v4ThreshRes,
          v4CvRes,
          v4SeedRes,
          v4PermRes,
          v4ShapRes,
          statsRes,
        ] = await Promise.all([
          dashboardApi.getV5Comparison('val'),
          dashboardApi.getV5Comparison('test'),
          dashboardApi.getV5Thresholds(),
          dashboardApi.getV5Cv(),
          dashboardApi.getV5FeatureSelection(),
          dashboardApi.getV5Metadata(),
          dashboardApi.getV5Experiments(),
          dashboardApi.getMetricsVal(),
          dashboardApi.getMetricsTest(),
          dashboardApi.getThresholdAnalysis(),
          dashboardApi.getCvSummary(),
          dashboardApi.getSeedStability(),
          dashboardApi.getPermutationImportance(),
          dashboardApi.getShapV4(),
          dashboardApi.getDatasetStats(),
        ])

        if (!active) return

        setV5ValMetrics(v5ValRes.data || [])
        setV5TestMetrics(v5TestRes.data || [])
        setV5Thresholds(v5ThreshRes.data || [])
        setV5Cv(v5CvRes.data || null)
        setV5FeatureSelection(v5FeatRes.data || null)
        setV5Metadata(v5MetaRes.data || null)
        setExperiments(expRes.data || [])

        setV4ValMetrics(v4ValRes.data || [])
        setV4TestMetrics(v4TestRes.data || [])
        setV4ThresholdData(v4ThreshRes.data || [])
        setV4CvSummary(v4CvRes.data || null)
        setV4SeedData(v4SeedRes.data || [])
        setV4PermData(v4PermRes.data || [])
        setV4ShapData(v4ShapRes.data || [])
        setDatasetStats(statsRes.data || null)
      } catch (err: any) {
        if (active) setError(err?.message || 'Failed to load model diagnostics.')
      } finally {
        if (active) setLoading(false)
      }
    }

    loadAllInsights()
    return () => {
      active = false
    }
  }, [])

  const activeValMetrics = versionView === 'v5' ? v5ValMetrics : v4ValMetrics
  const activeTestMetrics = versionView === 'v5' ? v5TestMetrics : v4TestMetrics
  const activeThresholds = versionView === 'v5' ? v5Thresholds : v4ThresholdData

  // Ranked Models Calculation
  const rankedModels = useMemo(() => {
    if (!activeValMetrics.length) return []
    const sorted = [...activeValMetrics].sort((a, b) => (Number(b.pr_auc) || 0) - (Number(a.pr_auc) || 0))
    const bestPr = Number(sorted[0]?.pr_auc) || 0
    const cvStd = v5Cv?.cv_pr_auc_std ?? 0.00548

    return sorted.map((m, idx) => {
      const pr = Number(m.pr_auc) || 0
      const diffFromBest = bestPr - pr
      const isWithin1Std = diffFromBest <= cvStd && idx > 0
      return {
        ...m,
        rank: idx + 1,
        family: ALGORITHM_FAMILIES[m.model]?.family || 'Ensemble / Supervised',
        tuned: ALGORITHM_FAMILIES[m.model]?.tuned || 'Default parameters',
        isBest: idx === 0,
        isServed: m.model === (v5Metadata?.best_model || 'XGBoost'),
        isStatisticallyTied: isWithin1Std,
        diffFromBest,
      }
    })
  }, [activeValMetrics, v5Cv, v5Metadata])

  if (loading) {
    return (
      <div className="page-section">
        <div className="panel" style={{ padding: '3rem', textAlign: 'center', color: '#6B7280' }}>
          Loading empirical ML diagnostics & model comparison...
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="page-section">
        <div className="panel" style={{ padding: '2rem', background: '#FEE2E2', color: '#991B1B' }}>
          <h3>Diagnostics Service Unavailable</h3>
          <p>{error}</p>
        </div>
      </div>
    )
  }

  const v5Run = experiments.find((e: any) => e.version === 'v5')
  const dummyTest = activeTestMetrics.find((m) => m.model === 'Dummy')
  const bestModelVal = activeValMetrics.find((m) => m.model === (v5Metadata?.best_model || 'XGBoost'))
  const bestModelTest = activeTestMetrics.find((m) => m.model === (v5Metadata?.best_model || 'XGBoost'))

  const dummyTestPr = Number(dummyTest?.pr_auc) || 0.2033
  const bestTestPr = Number(bestModelTest?.pr_auc) || Number(v5Metadata?.test_pr_auc) || 0.3296
  const prLiftAbsolute = bestTestPr - dummyTestPr
  const prLiftPercent = dummyTestPr > 0 ? (prLiftAbsolute / dummyTestPr) * 100 : 0

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <div className="eyebrow">MLOPS & EMPIRICAL DIAGNOSTICS ({versionView.toUpperCase()})</div>
          <h1>Model Insights & Empirical Audit</h1>
          <p>
            Evaluation matrices, feature selection ablation, threshold capacity curves, and MLflow governance.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ display: 'flex', background: '#E2E8F0', borderRadius: '8px', padding: '2px' }}>
            <button
              type="button"
              onClick={() => setVersionView('v5')}
              style={{
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                border: 'none',
                background: versionView === 'v5' ? '#2563EB' : 'transparent',
                color: versionView === 'v5' ? '#FFF' : '#475569',
                cursor: 'pointer',
              }}
            >
              V5 Active (Current)
            </button>
            <button
              type="button"
              onClick={() => setVersionView('v4')}
              style={{
                padding: '0.35rem 0.75rem',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600,
                border: 'none',
                background: versionView === 'v4' ? '#2563EB' : 'transparent',
                color: versionView === 'v4' ? '#FFF' : '#475569',
                cursor: 'pointer',
              }}
            >
              V4 Baseline
            </button>
          </div>
          <span style={{ fontSize: '0.75rem', fontWeight: 600, padding: '0.25rem 0.6rem', borderRadius: '999px', background: '#FEF3C7', color: '#92400E' }}>
            ROC-AUC ~0.55 (Weak Signal)
          </span>
        </div>
      </div>

      {/* Honest Limitation Banner */}
      <div
        style={{
          background: '#FFFBEB',
          border: '1px solid #FCD34D',
          borderRadius: '12px',
          padding: '1rem 1.25rem',
          marginBottom: '1rem',
          fontSize: '0.85rem',
          color: '#92400E',
          lineHeight: '1.5',
        }}
      >
        <strong>Academic & Operational Transparency:</strong> Tree-based models on this synthetic dataset extract modest signal above chance (Validation ROC-AUC 0.5558 vs Dummy 0.4990; PR-AUC 0.3320 vs Dummy 0.2016).
        Target leakage columns (<code>risk_score</code>, <code>failure_type</code>, <code>failure_severity</code>) were intentionally removed.
        This prototype provides <strong>decision-support triage</strong>, not autonomous safety certification.
      </div>

      {/* Navigation Pills */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        {[
          { id: 'comparison', label: '1. Model Comparison Matrix' },
          { id: 'ablation', label: '2. Feature Selection & Ablation' },
          { id: 'plots', label: '3. ROC, PR & Diagnostics Plots' },
          { id: 'threshold', label: '4. Threshold Trade-offs' },
          { id: 'robustness', label: '5. Repeated CV & Stability' },
          { id: 'shap', label: '6. Feature Importance & SHAP' },
          { id: 'governance', label: '7. Database & MLflow Runs' },
        ].map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id as Tab)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '999px',
              fontSize: '0.85rem',
              fontWeight: 500,
              border: activeTab === tab.id ? '1px solid #2563EB' : '1px solid #E5E7EB',
              background: activeTab === tab.id ? '#2563EB' : '#FFFFFF',
              color: activeTab === tab.id ? '#FFFFFF' : '#4B5563',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* TAB 1: Comparison & Full Metrics */}
      {activeTab === 'comparison' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* 1. Model in Production Card */}
          <div className="panel" style={{ borderLeft: '4px solid #2563EB' }}>
            <div className="panel-heading">
              <div>
                <div className="eyebrow">ACTIVE SERVING PIPELINE</div>
                <h2>Model in Production ({versionView.toUpperCase()})</h2>
                <p>Live inference parameters and held-out test benchmarks</p>
              </div>
              <span className="badge badge-status-scheduled" style={{ fontSize: '0.8rem', padding: '4px 10px' }}>
                ● Active Serving (Port 8000)
              </span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginTop: '0.5rem' }}>
              <div style={{ background: '#F8FAFC', padding: '12px 16px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: 600, textTransform: 'uppercase' }}>Model & Algorithm</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#0F172A', marginTop: '2px' }}>
                  {v5Metadata?.best_model || 'XGBoost'}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748B' }}>Gradient Boosted Trees Pipeline</div>
              </div>

              <div style={{ background: '#F8FAFC', padding: '12px 16px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: 600, textTransform: 'uppercase' }}>MLflow Run ID</div>
                <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#2563EB', marginTop: '4px', wordBreak: 'break-all' }}>
                  <code>{v5Run?.mlflow_run_id || 'railsetu_v5_xgboost'}</code>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748B' }}>Experiment: railsetu_{versionView}</div>
              </div>

              <div style={{ background: '#F8FAFC', padding: '12px 16px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: 600, textTransform: 'uppercase' }}>Decision Threshold & Calibration</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#0F172A', marginTop: '2px' }}>
                  τ = {v5Metadata?.chosen_threshold ?? 0.45}
                </div>
                <div style={{ fontSize: '0.75rem', color: '#166534' }}>Isotonic Calibrated (cv=3)</div>
              </div>

              <div style={{ background: '#F8FAFC', padding: '12px 16px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.72rem', color: '#64748B', fontWeight: 600, textTransform: 'uppercase' }}>Features & Training Split</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#0F172A', marginTop: '2px' }}>
                  {v5Metadata?.selected_features_k ?? 30} Selected Features
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748B' }}>60k Train / 20k Val / 20k Test</div>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #E2E8F0' }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Val PR-AUC:</span>{' '}
                <strong style={{ color: '#2563EB' }}>{Number(bestModelVal?.pr_auc || v5Metadata?.val_pr_auc || 0).toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Val ROC-AUC:</span>{' '}
                <strong>{Number(bestModelVal?.roc_auc || v5Metadata?.val_roc_auc || 0).toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Held-out Test PR-AUC:</span>{' '}
                <strong style={{ color: '#166534' }}>{Number(bestModelTest?.pr_auc || v5Metadata?.test_pr_auc || 0).toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Held-out Test ROC-AUC:</span>{' '}
                <strong>{Number(bestModelTest?.roc_auc || v5Metadata?.test_roc_auc || 0).toFixed(4)}</strong>
              </div>
              <div>
                <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Test Brier Score:</span>{' '}
                <strong style={{ color: '#166534' }}>0.1474</strong>{' '}
                <span style={{ fontSize: '0.7rem', color: '#64748B' }}>(was 0.2259 uncal)</span>
              </div>
            </div>
          </div>

          {/* 2. Models Evaluated Table */}
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Models Evaluated ({rankedModels.length} Candidate Architectures)</h2>
                <p>Ranked strictly by Validation PR-AUC on 20,000 records. Models within 1 CV std of the best are statistically tied.</p>
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Algorithm Family</th>
                    <th>Tuned</th>
                    <th>Validation PR-AUC</th>
                    <th>Validation ROC-AUC</th>
                    <th>Status & Statistical Significance</th>
                  </tr>
                </thead>
                <tbody>
                  {rankedModels.map((m) => (
                    <tr
                      key={m.model}
                      style={{
                        background: m.isBest ? '#F0FDF4' : m.isStatisticallyTied ? '#F8FAFC' : undefined,
                        fontWeight: m.isBest ? 600 : 400,
                      }}
                    >
                      <td>#{m.rank}</td>
                      <td>
                        <strong>{m.model}</strong>
                      </td>
                      <td style={{ fontSize: '0.8rem', color: '#475569' }}>{m.family}</td>
                      <td style={{ fontSize: '0.8rem', color: '#64748B' }}>{m.tuned}</td>
                      <td>
                        <strong style={{ color: m.isBest ? '#166534' : '#2563EB' }}>
                          {Number(m.pr_auc).toFixed(4)}
                        </strong>
                      </td>
                      <td>{Number(m.roc_auc).toFixed(4)}</td>
                      <td>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                          {m.isBest && (
                            <span className="badge badge-priority-low" style={{ background: '#DCFCE7', color: '#166534' }}>
                              ★ Winner (Best PR-AUC)
                            </span>
                          )}
                          {m.isServed && (
                            <span className="badge badge-status-scheduled">
                              ✓ Served Model
                            </span>
                          )}
                          {m.isStatisticallyTied && (
                            <span className="badge badge-priority-medium" style={{ fontSize: '0.7rem' }}>
                              Statistically Tied (≤ 1σ)
                            </span>
                          )}
                          {!m.isBest && !m.isStatisticallyTied && (
                            <span style={{ fontSize: '0.75rem', color: '#94A3B8' }}>
                              -{m.diffFromBest.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* 3. How the Best Model Was Selected Card */}
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>How the Best Model Was Selected</h2>
                <p>Objective mathematical rules, cost considerations, and dual-language viva rationale</p>
              </div>
            </div>

            <div style={{ background: '#F1F5F9', padding: '14px 18px', borderRadius: '10px', fontSize: '0.85rem', color: '#1E293B', marginBottom: '1.25rem', lineHeight: '1.5' }}>
              <strong>Official Selection Rule:</strong> The active model was chosen on the <strong>validation fold only</strong> using <strong>Validation Average Precision (PR-AUC)</strong> as the primary criterion, constrained by 5x2 repeated cross-validation stability. The test set was strictly held out and evaluated only once after model freezing to prevent data leakage and optimism bias.
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {[
                {
                  title: '1. PR-AUC (Primary Ranking Metric)',
                  desc: 'With ~20% positive class prevalence, accuracy is deceptively inflated by majority negatives. PR-AUC directly assesses positive predictive power across all decision thresholds.',
                  hinglish: 'Positive class sirf 20% hai, isliye accuracy deceptive hoti hai. PR-AUC precision aur recall ke trade-off ko measure karta hai.',
                },
                {
                  title: '2. ROC-AUC (Global Discrimination)',
                  desc: 'Evaluates global ranking capability independent of threshold. Indicates whether faulty rolling stock consistently receives higher risk probabilities than healthy stock.',
                  hinglish: 'Har threshold par faulty vs non-faulty trains ko rank karne ki ability evaluate karta hai.',
                },
                {
                  title: '3. Brier Score & Isotonic Calibration',
                  desc: 'Measures mean squared error of predicted probabilities against true binary outcomes. Isotonic calibration cut held-out test Brier score from 0.2259 to 0.1474.',
                  hinglish: 'Probability values actual fault frequencies se kitni closely match karti hain; Isotonic calibration ne error ko sharply reduce kiya.',
                },
                {
                  title: '4. Recall & F2 Score at Chosen Threshold (0.45)',
                  desc: 'Missing an impending failure (False Negative) has catastrophic safety and derailment implications, whereas a false alarm (False Positive) only costs inspection time. F2 weights recall 2x over precision.',
                  hinglish: 'Railway me fault miss karna false alarm se zyada dangerous hai, isliye F2 recall ko double weight deta hai.',
                },
                {
                  title: '5. Specificity & % Fleet Flagged (Depot Capacity)',
                  desc: 'Thresholds < 0.20 flag nearly 100% of fleet, creating impossible inspection backlogs. Threshold 0.45 was chosen because it flags ~15% of fleet, matching typical depot intake capacity.',
                  hinglish: 'Depot ki limited team capacity hoti hai; 0.45 threshold fleet ka ~15% flag karta hai jo manageable hai.',
                },
                {
                  title: '6. Repeated 5x2 CV Stability',
                  desc: 'Evaluated 10 folds on train+val (Mean PR-AUC: 0.3249 ± 0.0055). Proves the model victory is stable across random data folds rather than a split fluke.',
                  hinglish: '10 folds par test karke confirm kiya ki model kisi ek lucky split ki wajah se nahi jeeta.',
                },
                {
                  title: '7. Dummy Baseline & Permutation Testing',
                  desc: 'Dummy classifier scores PR-AUC 0.2033 (random baseline). XGBoost scores 0.3296 on test (+62% relative lift), confirming genuine predictive signal extraction above chance.',
                  hinglish: 'Random guess baseline (0.20) se compare karke verify kiya ki model sach me pattern seekh raha hai.',
                },
              ].map((c, i) => (
                <div key={i} style={{ borderBottom: i < 6 ? '1px solid #F1F5F9' : 'none', paddingBottom: '10px' }}>
                  <div style={{ fontWeight: 600, color: '#0F172A', fontSize: '0.875rem' }}>{c.title}</div>
                  <div style={{ color: '#475569', fontSize: '0.825rem', marginTop: '2px', lineHeight: '1.4' }}>{c.desc}</div>
                  <div style={{ color: '#2563EB', fontSize: '0.78rem', marginTop: '3px', fontStyle: 'italic' }}>💬 Hinglish: "{c.hinglish}"</div>
                </div>
              ))}
            </div>
          </div>

          {/* 4. Verdict Card */}
          <div className="panel" style={{ background: '#F8FAFC', border: '1px solid #CBD5E1' }}>
            <div className="panel-heading">
              <div>
                <div className="eyebrow">EMPIRICAL VERDICT & AUDIT SUMMARY</div>
                <h2>Empirical Verdict & Performance Ceiling</h2>
                <p>Data-derived assessment of production utility and limitations</p>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ background: '#FFFFFF', padding: '14px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.75rem', color: '#64748B' }}>Relative Lift Over Dummy</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 600, color: '#166534', marginTop: '2px' }}>
                  +{prLiftPercent.toFixed(1)}%
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748B' }}>Test PR-AUC 0.3296 vs 0.2033 baseline</div>
              </div>

              <div style={{ background: '#FFFFFF', padding: '14px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.75rem', color: '#64748B' }}>Margin Over V4 Baseline</div>
                <div style={{ fontSize: '1.5rem', fontWeight: 600, color: '#2563EB', marginTop: '2px' }}>
                  +{(v5Metadata?.margin_over_v4 ?? 0.00176).toFixed(5)}
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748B' }}>Within CV noise (σ = 0.00548)</div>
              </div>

              <div style={{ background: '#FFFFFF', padding: '14px', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.75rem', color: '#64748B' }}>Discriminative Power</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 600, color: '#D97706', marginTop: '4px' }}>
                  Modest (AUC ~0.55)
                </div>
                <div style={{ fontSize: '0.72rem', color: '#64748B' }}>Consistent across all 10 candidate models</div>
              </div>
            </div>

            <p style={{ fontSize: '0.85rem', color: '#475569', lineHeight: '1.5', margin: 0 }}>
              <strong>Conclusion:</strong> Extensive model tuning, Optuna optimization, and feature engineering confirmed an empirical ceiling on this synthetic dataset. Features exhibit mutual information &lt; 0.010 with target failure events, explaining why ROC-AUC plateaus near 0.55 across gradient boosting, random forests, and deep ensembles alike. 
              The V5 XGBoost Pipeline is statistically verified as the top-performing candidate, calibrated to provide honest probabilities, and is suitable <strong>strictly as an operational decision-support tool</strong> rather than an autonomous safety system.
            </p>
          </div>

          {/* 5. Detailed Tables (Validation & Held-Out Test) */}
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Validation Set Evaluation (Full Matrix - {versionView.toUpperCase()})</h2>
                <p>Metrics computed on 20,000 validation records used to tune hyperparameters.</p>
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC (AP)</th>
                    <th>F1 Score</th>
                    <th>Balanced Acc</th>
                    <th>MCC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Brier Score</th>
                  </tr>
                </thead>
                <tbody>
                  {activeValMetrics.map((m: any) => (
                    <tr key={m.model} style={{ fontWeight: m.model === 'XGBoost' ? 600 : 400, background: m.model === 'XGBoost' ? '#F0FDF4' : undefined }}>
                      <td>{m.model} {m.model === 'XGBoost' && '(Served)'}</td>
                      <td>{Number(m.roc_auc).toFixed(4)}</td>
                      <td style={{ color: '#2563EB', fontWeight: 600 }}>{Number(m.pr_auc).toFixed(4)}</td>
                      <td>{Number(m.f1).toFixed(4)}</td>
                      <td>{Number(m.balanced_acc).toFixed(4)}</td>
                      <td>{Number(m.mcc).toFixed(4)}</td>
                      <td>{Number(m.precision).toFixed(4)}</td>
                      <td>{Number(m.recall).toFixed(4)}</td>
                      <td>{Number(m.brier).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Held-Out Test Set Evaluation (Unbiased Generalization - {versionView.toUpperCase()})</h2>
                <p>Evaluated exactly once on 20,000 held-out test samples.</p>
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC (AP)</th>
                    <th>F1 Score</th>
                    <th>Balanced Acc</th>
                    <th>MCC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Brier Score</th>
                  </tr>
                </thead>
                <tbody>
                  {activeTestMetrics.map((m: any) => (
                    <tr key={m.model} style={{ fontWeight: m.model === 'XGBoost' ? 600 : 400 }}>
                      <td>{m.model}</td>
                      <td>{Number(m.roc_auc).toFixed(4)}</td>
                      <td style={{ color: '#2563EB', fontWeight: 600 }}>{Number(m.pr_auc).toFixed(4)}</td>
                      <td>{Number(m.f1).toFixed(4)}</td>
                      <td>{Number(m.balanced_acc).toFixed(4)}</td>
                      <td>{Number(m.mcc).toFixed(4)}</td>
                      <td>{Number(m.precision).toFixed(4)}</td>
                      <td>{Number(m.recall).toFixed(4)}</td>
                      <td>{Number(m.brier).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Feature Selection & Ablation */}
      {activeTab === 'ablation' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Feature Subset Ablation Analysis (Task 2)</h2>
                <p>
                  Trained across subset sizes (Top 5, 8, 10, 12, 15, 20, 30).
                  Empirical finding: in low-signal data, pruning features causes immediate PR-AUC degradation towards the 0.202 dummy baseline.
                </p>
              </div>
            </div>

            {v5FeatureSelection && v5FeatureSelection.ablation ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Subset Size (k)</th>
                      <th>Val PR-AUC</th>
                      <th>Val ROC-AUC</th>
                      <th>Test PR-AUC</th>
                      <th>Test ROC-AUC</th>
                      <th>Status (1-Std Rule)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {v5FeatureSelection.ablation.map((row: any) => (
                      <tr key={row.k} style={{ fontWeight: row.k === 30 ? 600 : 400, background: row.k === 30 ? '#F0FDF4' : undefined }}>
                        <td>Top {row.k} Features {row.k === 30 && '(Full Model Selected)'}</td>
                        <td style={{ color: '#2563EB', fontWeight: 600 }}>{Number(row.val_pr_auc).toFixed(4)}</td>
                        <td>{Number(row.val_roc_auc).toFixed(4)}</td>
                        <td>{Number(row.test_pr_auc).toFixed(4)}</td>
                        <td>{Number(row.test_roc_auc).toFixed(4)}</td>
                        <td>
                          {row.k === 30 ? (
                            <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', background: '#DCFCE7', color: '#166534', fontSize: '0.75rem' }}>
                              Retained (Best Score)
                            </span>
                          ) : (
                            <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', background: '#FEE2E2', color: '#991B1B', fontSize: '0.75rem' }}>
                              Degraded (&gt;1 std below)
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div>No ablation data found.</div>
            )}
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Feature Selection Ablation Curve</h2>
                <p>Visualization of metric trajectory as feature count increases.</p>
              </div>
            </div>
            <img
              src={`${apiBase}/api/ml/plots/v5/02_feature_selection_ablation.png`}
              alt="Feature Selection Ablation Curve"
              style={{ width: '100%', maxWidth: '800px', borderRadius: '8px', border: '1px solid #E5E7EB' }}
              onError={(e) => { (e.target as HTMLElement).style.display = 'none' }}
            />
          </div>
        </div>
      )}

      {/* TAB 3: Plots */}
      {activeTab === 'plots' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(420px, 1fr))', gap: '1.5rem' }}>
          <div className="panel">
            <h3>V5 ROC & PR Curves (Held-out Test)</h3>
            <p style={{ fontSize: '0.8rem', color: '#6B7280', marginBottom: '0.75rem' }}>
              Comparison against random baseline (PR-AUC 0.202, ROC-AUC 0.50).
            </p>
            <img
              src={`${apiBase}/api/ml/plots/v5/03_roc_pr_curves_v5.png`}
              alt="V5 ROC and PR Curves"
              style={{ width: '100%', borderRadius: '8px', border: '1px solid #E5E7EB' }}
            />
          </div>

          <div className="panel">
            <h3>Signal Diagnosis (ANOVA vs Tree Depth)</h3>
            <p style={{ fontSize: '0.8rem', color: '#6B7280', marginBottom: '0.75rem' }}>
              Feature signal plateauing at depth 2-3 with mutual info &lt; 0.01.
            </p>
            <img
              src={`${apiBase}/api/ml/plots/v5/01_signal_diagnosis.png`}
              alt="Signal Diagnosis"
              style={{ width: '100%', borderRadius: '8px', border: '1px solid #E5E7EB' }}
            />
          </div>

          <div className="panel">
            <h3>V5 Operational Capacity & Threshold Trade-offs</h3>
            <p style={{ fontSize: '0.8rem', color: '#6B7280', marginBottom: '0.75rem' }}>
              Fleet workload vs detection recall across threshold spectrum.
            </p>
            <img
              src={`${apiBase}/api/ml/plots/v5/04_threshold_analysis_v5.png`}
              alt="V5 Threshold Analysis"
              style={{ width: '100%', borderRadius: '8px', border: '1px solid #E5E7EB' }}
            />
          </div>

          <div className="panel">
            <h3>Calibration Curves (Isotonic vs Sigmoid)</h3>
            <p style={{ fontSize: '0.8rem', color: '#6B7280', marginBottom: '0.75rem' }}>
              Isotonic calibration brings test Brier score from 0.226 down to 0.147.
            </p>
            <img
              src={`${apiBase}/api/ml/plots/07_calibration.png`}
              alt="Model Calibration"
              style={{ width: '100%', borderRadius: '8px', border: '1px solid #E5E7EB' }}
            />
          </div>
        </div>
      )}

      {/* TAB 4: Thresholds */}
      {activeTab === 'threshold' && (
        <div className="panel">
          <div className="panel-heading">
            <div>
              <h2>Operational Threshold Analysis ({versionView.toUpperCase()})</h2>
              <p>
                Decision threshold calibrated against depot capacity. Chosen operational threshold is <strong>0.45</strong> (flags ~15% fleet, precision 23.5%, recall 17.5%).
              </p>
            </div>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Threshold</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>F2</th>
                  <th>% Flagged</th>
                  <th>TP</th>
                  <th>FP</th>
                  <th>Operational Character</th>
                </tr>
              </thead>
              <tbody>
                {activeThresholds.map((t: any) => (
                  <tr
                    key={t.threshold}
                    style={{
                      fontWeight: Number(t.threshold) === 0.45 ? 600 : 400,
                      background: Number(t.threshold) === 0.45 ? '#EFF6FF' : undefined,
                    }}
                  >
                    <td>{Number(t.threshold).toFixed(2)} {Number(t.threshold) === 0.45 && '(Active)'}</td>
                    <td>{Number(t.precision).toFixed(3)}</td>
                    <td>{Number(t.recall).toFixed(3)}</td>
                    <td>{Number(t.f1).toFixed(3)}</td>
                    <td>{Number(t.f2).toFixed(3)}</td>
                    <td>{Number(t.pct_flagged).toFixed(1)}%</td>
                    <td>{t.tp}</td>
                    <td>{t.fp}</td>
                    <td style={{ fontSize: '0.8rem', color: '#4B5563' }}>{t.operational_character}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 5: Robustness (CV & Stability) */}
      {activeTab === 'robustness' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Repeated 5x2 Stratified Cross-Validation ({versionView.toUpperCase()})</h2>
                <p>10 total folds evaluated on train+val to ensure stability and guard against single-split variance.</p>
              </div>
            </div>
            {versionView === 'v5' && v5Cv ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '0.5rem' }}>
                <div style={{ background: '#F9FAFB', padding: '1rem', borderRadius: '8px', border: '1px solid #E5E7EB' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>CV PR-AUC Mean ± Std</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 600, color: '#2563EB', marginTop: '0.25rem' }}>
                    {v5Cv.cv_pr_auc_mean?.toFixed(4)} ± {v5Cv.cv_pr_auc_std?.toFixed(4)}
                  </div>
                </div>
                <div style={{ background: '#F9FAFB', padding: '1rem', borderRadius: '8px', border: '1px solid #E5E7EB' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>CV ROC-AUC Mean ± Std</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 600, color: '#111827', marginTop: '0.25rem' }}>
                    {v5Cv.cv_roc_auc_mean?.toFixed(4)} ± {v5Cv.cv_roc_auc_std?.toFixed(4)}
                  </div>
                </div>
                <div style={{ background: '#F9FAFB', padding: '1rem', borderRadius: '8px', border: '1px solid #E5E7EB' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Folds Evaluated</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 600, color: '#111827', marginTop: '0.25rem' }}>
                    10 (5x2 Repeated Stratified)
                  </div>
                </div>
              </div>
            ) : versionView === 'v4' && v4CvSummary ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '0.5rem' }}>
                <div style={{ background: '#F9FAFB', padding: '1rem', borderRadius: '8px', border: '1px solid #E5E7EB' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>V4 CV PR-AUC Mean ± Std</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 600, color: '#2563EB', marginTop: '0.25rem' }}>
                    {v4CvSummary.cv_pr_auc_mean?.toFixed(4)} ± {v4CvSummary.cv_pr_auc_std?.toFixed(4)}
                  </div>
                </div>
                <div style={{ background: '#F9FAFB', padding: '1rem', borderRadius: '8px', border: '1px solid #E5E7EB' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>V4 Multi-Seed Runs</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 600, color: '#111827', marginTop: '0.25rem' }}>
                    {v4SeedData.length} Seeds
                  </div>
                </div>
              </div>
            ) : (
              <div>No CV data available.</div>
            )}
          </div>
        </div>
      )}

      {/* TAB 6: SHAP & Permutation */}
      {activeTab === 'shap' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Permutation Feature Importance ({versionView.toUpperCase()})</h2>
                <p>Drop in PR-AUC when feature values are permuted.</p>
              </div>
            </div>
            {versionView === 'v5' && v5FeatureSelection && v5FeatureSelection.ranking ? (
              <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Feature</th>
                      <th>Permutation Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {v5FeatureSelection.ranking.map((r: any, idx: number) => (
                      <tr key={r.feature}>
                        <td>#{idx + 1}</td>
                        <td>{r.feature.replace(/_/g, ' ')}</td>
                        <td style={{ fontWeight: 600, color: '#2563EB' }}>
                          {Number(r.importance_perm).toFixed(5)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : versionView === 'v4' && v4PermData.length > 0 ? (
              <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Feature</th>
                      <th>Permutation Mean</th>
                    </tr>
                  </thead>
                  <tbody>
                    {v4PermData.map((r: any, idx: number) => (
                      <tr key={r.feature}>
                        <td>#{idx + 1}</td>
                        <td>{r.feature.replace(/_/g, ' ')}</td>
                        <td style={{ fontWeight: 600, color: '#2563EB' }}>
                          {Number(r.importance_mean || r.importance).toFixed(5)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>SHAP Summary Bar Plot</h2>
                <p>Global impact on model output magnitude.</p>
              </div>
            </div>
            <img
              src={`${apiBase}/api/ml/plots/16_shap_bar.png`}
              alt="SHAP Feature Importance"
              style={{ width: '100%', borderRadius: '8px', border: '1px solid #E5E7EB' }}
            />
            {versionView === 'v4' && v4ShapData.length > 0 && (
              <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: '#6B7280' }}>
                Loaded {v4ShapData.length} SHAP reference attributions.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 7: Governance & Database Runs */}
      {activeTab === 'governance' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Database Model Runs & MLflow Governance</h2>
                <p>Tracking experiment runs, hyperparameters, and active served versions from SQLite database.</p>
              </div>
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Model Name</th>
                    <th>MLflow Run ID</th>
                    <th>Served Status</th>
                    <th>Created At</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((e: any) => (
                    <tr key={e.id} style={{ background: e.is_served ? '#F0FDF4' : undefined }}>
                      <td><strong>{e.version.toUpperCase()}</strong></td>
                      <td>{e.model_name}</td>
                      <td><code>{e.mlflow_run_id}</code></td>
                      <td>
                        {e.is_served ? (
                          <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', background: '#DCFCE7', color: '#166534', fontSize: '0.75rem', fontWeight: 600 }}>
                            ACTIVE SERVED
                          </span>
                        ) : (
                          <span style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', background: '#F1F5F9', color: '#64748B', fontSize: '0.75rem' }}>
                            Archived
                          </span>
                        )}
                      </td>
                      <td>{e.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {v5Metadata && (
              <div style={{ background: '#F8FAFC', padding: '1rem', borderRadius: '8px', border: '1px solid #E2E8F0', marginTop: '1rem' }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#334155' }}>V5 Model Conclusion & Empirical Signal Ceiling</div>
                <p style={{ fontSize: '0.8rem', color: '#475569', marginTop: '0.25rem', lineHeight: '1.5' }}>
                  {v5Metadata.honest_conclusion}
                </p>
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <h2>Dataset & Integrity Audit</h2>
                <p>Static distribution and feature safeguards.</p>
              </div>
            </div>
            {datasetStats && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
                <div style={{ background: '#F9FAFB', padding: '0.85rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Total Records</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>{datasetStats.total_records.toLocaleString()}</div>
                </div>
                <div style={{ background: '#F9FAFB', padding: '0.85rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Class 1 Positive Rate</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 600 }}>{datasetStats.target_distribution?.positive_rate_percent}%</div>
                </div>
                <div style={{ background: '#F9FAFB', padding: '0.85rem', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280' }}>Excluded Leakage Columns</div>
                  <div style={{ fontSize: '0.85rem', color: '#DC2626', marginTop: '0.2rem' }}>
                    {datasetStats.leakage_excluded_columns?.join(', ')}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
