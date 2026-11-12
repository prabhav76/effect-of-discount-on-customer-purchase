import { useEffect, useState } from 'react'
import {
  ArrowRight, BarChart3, CircleHelp, Download, LayoutDashboard, Menu,
  ShoppingBasket, Sparkles, Tag, TrendingUp, Users, X,
} from 'lucide-react'
import './App.css'
import './project.css'
import './presentation.css'

const pages = [
  { id: 'Overview', icon: LayoutDashboard },
  { id: 'Discount & quantity', icon: Tag },
  { id: 'Promotion effectiveness', icon: BarChart3 },
  { id: 'Household clusters', icon: Users },
  { id: 'Basket associations', icon: ShoppingBasket },
]

const percent = (value, digits = 2) => `${Number(value || 0).toFixed(digits)}%`
const odds = (value, digits = 2) => `${Number(value || 0).toFixed(digits)}×`
const money = (value) => `$${Number(value || 0).toFixed(2)}`
const whole = (value) => Number(value || 0).toLocaleString()
const pValue = (value) => Number(value) < 0.001 ? 'p < 0.001' : `p = ${Number(value).toFixed(3)}`

function bandTrend(bands) {
  const valid = bands.filter(band => Number.isFinite(band.rate))
  if (valid.length < 2) return { label: 'Insufficient data', highest: null }
  const differences = valid.slice(1).map((band, index) => band.rate - valid[index].rate)
  const nondecreasing = differences.every(value => value >= 0)
  const nonincreasing = differences.every(value => value <= 0)
  return {
    label: nondecreasing || nonincreasing ? 'Dose-related' : 'Non-linear',
    highest: valid.reduce((best, band) => band.rate > best.rate ? band : best),
  }
}

function Logo() {
  return <div className="brand-lockup"><div className="brand-mark"><span /><span /><span /></div><span className="brand-name">Purchase<span className="brand-dot"> patterns</span></span></div>
}

function StatCard({ label, value, note, icon: Icon, color = 'sage' }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className={`stat-icon ${color}`}><Icon size={18} strokeWidth={1.8} /></span></div><div className="stat-value">{value}</div><div className="stat-note">{note}</div></article>
}

function PageHeader({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>
}

function App() {
  const [page, setPage] = useState('Overview')
  const [mobileNav, setMobileNav] = useState(false)
  const [data, setData] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [toast, setToast] = useState('')
  const notify = (message) => { setToast(message); window.setTimeout(() => setToast(''), 3500) }

  useEffect(() => {
    let cancelled = false
    fetch(`/data/dashboard.json?refresh=${Date.now()}`, { cache: 'no-store' })
      .then(response => {
        if (!response.ok) throw new Error('Dashboard data could not be loaded.')
        return response.json()
      })
      .then(payload => {
        if (cancelled) return
        if (!payload.available) throw new Error(payload.message || 'Analysis results are not available.')
        setData(payload)
      })
      .catch(error => { if (!cancelled) setLoadError(error.message) })
    return () => { cancelled = true }
  }, [])

  const exportCsv = () => {
    if (!data) return
    const rows = [['Analysis area', 'Finding', 'Method / note'], ...data.findings.map(row => [row.area, row.finding, row.method])]
    const csv = rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    const link = document.createElement('a')
    link.href = url
    link.download = 'discount-purchase-analysis-findings.csv'
    link.click()
    URL.revokeObjectURL(url)
    notify('Analysis findings exported as CSV.')
  }

  const action = <button className="button button-dark" onClick={exportCsv}><Download size={17} /><span>Export findings</span></button>
  const promoFindings = data?.promotion.findings || []
  const basketSignificant = Number(data?.promotion.basket_sales_significant) + Number(data?.promotion.basket_discount_significant)
  const couponSignificant = Number(data?.promotion.coupon_sales_significant) + Number(data?.promotion.coupon_discount_significant)
  const clusters = data?.clusters.profiles || []
  const discountTrend = data ? bandTrend(data.discount.bands) : { label: '—', highest: null }

  return <div className="app-shell min-h-screen bg-paper text-ink antialiased">
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
      <div className="sidebar-top"><Logo /><button className="mobile-close icon-button" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={20} /></button></div>
      <div className="workspace-switch"><div className="workspace-avatar">DP</div><div className="workspace-copy"><b>Grocery purchase study</b><span>Discount & promotion analysis</span></div></div>
      <div className="nav-caption">PROJECT ANALYSIS</div>
      <nav className="main-nav" aria-label="Project analysis navigation">{pages.map(({ id, icon: Icon }) => <button key={id} onClick={() => { setPage(id); setMobileNav(false) }} className={`nav-item ${page === id ? 'active' : ''}`}><Icon size={19} strokeWidth={1.8} /><span>{id}</span></button>)}</nav>
      <div className="sidebar-divider" />
      <div className="nav-caption">STUDY SCOPE</div>
      <div className="scope-item"><i className="scope-dot green-dot" /><span>Grocery transactions</span></div>
      <div className="scope-item"><i className="scope-dot orange-dot" /><span>{data ? `${whole(data.overview.households)} households` : 'Household analysis'}</span></div>
      <div className="sidebar-spacer" />
      <div className="sidebar-insight"><div className="insight-icon"><CircleHelp size={17} /></div><div><b>Study interpretation</b><p>Results describe associations in observed purchases. The study design cannot establish that discounts caused a purchase.</p></div></div>
      <div className="sidebar-footnote">Grocery retail · household purchase behavior</div>
    </aside>

    <main className="main-content">
      <header className="topbar"><button className="mobile-menu icon-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={21} /></button><div className="breadcrumb">Research <span>/</span> <b>{page}</b></div><div className="topbar-right">{data && <div className="data-status"><i /> {whole(data.overview.households)} households</div>}</div></header>
      <div className="content-area">
        {!data && !loadError && <section className="data-message"><span className="loading-mark" /><h1>Loading analysis results</h1><p>Preparing the latest project findings.</p></section>}
        {loadError && <section className="data-message error"><h1>Analysis results unavailable</h1><p>{loadError}</p><p>Run <code>npm run refresh-data</code> from the frontend folder after generating the processed analysis files.</p></section>}
        {data && <>
          {page === 'Overview' && <>
            <PageHeader eyebrow="GROCERY RETAIL · OBSERVATIONAL STUDY" title="Discounts & purchase behavior" description="How do discounts relate to purchase quantity, promotion outcomes, household patterns, and products bought together?" action={action} />
            <div className="snapshot-note"><span className="snapshot-dot" /><span><b>Study measures</b> — a multi-unit purchase has quantity greater than one; discounted records have a positive recorded discount amount.</span></div>
            <div className="stat-grid">
              <StatCard label="Households in statistical analysis" value={whole(data.overview.households)} note="Household-level clustering used for inference" icon={Users} color="sage" />
              <StatCard label="Multi-unit rate · discounted" value={percent(data.overview.discounted_multi_unit_rate * 100)} note="Discounted transaction records" icon={Tag} color="peach" />
              <StatCard label="Multi-unit rate · no discount" value={percent(data.overview.non_discounted_multi_unit_rate * 100)} note="Non-discounted transaction records" icon={ShoppingBasket} color="blue" />
              <StatCard label="Observed difference" value={`${Number(data.overview.difference_pp).toFixed(2)} pp`} note={`95% household-bootstrap CI: ${Number(data.overview.bootstrap_ci_lower).toFixed(2)}–${Number(data.overview.bootstrap_ci_upper).toFixed(2)} pp`} icon={TrendingUp} color="lavender" />
            </div>
            <div className="lower-grid overview-grid">
              <section className="panel category-panel"><div className="panel-heading"><div><h2>Discount status and multi-unit purchases</h2><p>Share of transaction records with quantity greater than one</p></div></div><div className="rate-comparison">{[{ label: 'Discounted', value: data.overview.discounted_multi_unit_rate * 100, tone: 'green' }, { label: 'No discount', value: data.overview.non_discounted_multi_unit_rate * 100, tone: 'orange' }].map(item => <div className="rate-row" key={item.label}><div className="rate-label"><b>{item.label}</b><span>{percent(item.value)}</span></div><div className="rate-track"><i className={item.tone} style={{ width: `${item.value * 3}%` }} /></div></div>)}</div><div className="result-callout"><b>{odds(data.overview.cmh_odds_ratio)}</b><span>Higher odds of multi-unit purchasing for discounted records after accounting for household differences ({pValue(data.overview.cmh_p_value)}).</span></div></section>
              <section className="panel segment-panel"><div className="panel-heading"><div><h2>Study overview</h2><p>Four complementary views of purchase behavior</p></div></div><div className="coverage-list"><div><span>Discount and purchase quantity</span><b>{data.discount.category_count ? '4 questions' : '—'}</b></div><div><span>Promotion and coupon outcomes</span><b>{whole(data.promotion.test_count)} comparisons</b></div><div><span>Household purchase patterns</span><b>{whole(clusters.length)} groups</b></div><div><span>Products bought together</span><b>{whole(data.basket.rule_count)} rules</b></div></div></section>
            </div>
            <div className="insight-banner"><div className="banner-icon"><Sparkles size={18} /></div><div><b>Discount association is clear; the relationship is {discountTrend.label.toLowerCase()}</b><span>{discountTrend.highest?.name || 'The leading'} discount band has the highest observed multi-unit purchase rate.</span></div><button onClick={() => setPage('Discount & quantity')}>Explore results <ArrowRight size={16} /></button></div>
          </>}

          {page === 'Discount & quantity' && <>
            <PageHeader eyebrow="RESEARCH QUESTIONS 1–4" title="Discounts & purchase quantity" description="Household bootstrap and stratified tests account for repeated purchases; GEE models compare discount bands, brands, and categories." action={action} />
            <div className="stat-grid"><StatCard label="Discounted multi-unit rate" value={percent(data.overview.discounted_multi_unit_rate * 100)} note="Transaction records with a recorded discount" icon={Tag} color="sage" /><StatCard label="Non-discounted multi-unit rate" value={percent(data.overview.non_discounted_multi_unit_rate * 100)} note="Transaction records without a discount" icon={ShoppingBasket} color="peach" /><StatCard label="Household-adjusted odds ratio" value={Number(data.overview.cmh_odds_ratio).toFixed(3)} note={`CMH 95% CI: ${Number(data.overview.cmh_ci_lower).toFixed(3)}–${Number(data.overview.cmh_ci_upper).toFixed(3)} · ${pValue(data.overview.cmh_p_value)}`} icon={BarChart3} color="blue" /><StatCard label="Discount-band pattern" value={discountTrend.label} note={`${discountTrend.highest?.name || 'Leading'} band had the highest rate`} icon={TrendingUp} color="lavender" /></div>
            <section className="panel category-panel analysis-panel"><div className="panel-heading"><div><h2>Multi-unit rate by discount band</h2><p>Observed rates from the enriched transaction records</p></div></div><div className="rate-comparison band-comparison">{data.discount.bands.map(item => <div className="rate-row" key={item.name}><div className="rate-label"><div><b>{item.name}</b><small>{whole(item.records)} transaction records</small></div><span>{item.rate === null ? '—' : percent(item.rate)}</span></div><div className="rate-track"><i className={item.name === discountTrend.highest?.name ? 'green' : 'orange'} style={{ width: `${Math.min(100, (item.rate || 0) * 2.5)}%` }} /></div></div>)}</div><div className="stats-inline"><div><b>{odds(data.discount.low_vs_no_discount_or)}</b><span>Low vs no-discount odds</span></div><div><b>{odds(data.discount.very_high_vs_no_discount_or)}</b><span>Very-high vs no-discount odds</span></div><div><b>{pValue(data.discount.overall_band_p_value)}</b><span>Overall discount-band association</span></div></div></section>
            <div className="lower-grid"><section className="panel category-panel"><div className="panel-heading"><div><h2>Brand comparison</h2><p>Discount association with multi-unit purchasing</p></div></div><div className="coverage-list"><div><span>National brand odds ratio</span><b>{Number(data.discount.national_or).toFixed(2)}</b></div><div><span>Private brand odds ratio</span><b>{Number(data.discount.private_or).toFixed(2)}</b></div><div><span>Brand interaction</span><b>p = {Number(data.discount.brand_interaction_p_value).toFixed(3)}</b></div></div><p className="panel-footnote">The association was slightly stronger for Private brands; the observational design cannot establish a causal effect.</p></section><section className="panel category-panel"><div className="panel-heading"><div><h2>Category-level results</h2><p>Household-clustered category comparisons</p></div></div><div className="result-callout category-callout"><b>{whole(data.discount.significant_category_count)}</b><span>of {whole(data.discount.category_count)} eligible categories remained significant after Holm correction. {whole(data.discount.positive_category_count)} were positive and {whole(data.discount.negative_category_count)} negative.</span></div><p className="panel-footnote">Median category odds ratio: {Number(data.discount.category_median_or).toFixed(2)}. Results vary across categories.</p></section></div>
          </>}

          {page === 'Promotion effectiveness' && <>
            <PageHeader eyebrow="PROMOTION & COUPON ANALYSIS" title="Promotion & coupon results" description="Compare recorded promotional exposure and coupon use at household and basket level." action={action} />
            <div className="stat-grid"><StatCard label="Records with promotion metadata" value={percent(data.promotion.metadata_share_pct)} note="Descriptive exposure coverage" icon={Tag} color="sage" /><StatCard label="Basket-level outcomes" value={`${basketSignificant} significant`} note="Sales and discount percentage after Holm correction" icon={ShoppingBasket} color="peach" /><StatCard label="Coupon household tests" value={`${couponSignificant} significant`} note="Coupon usage and household outcomes" icon={Users} color="blue" /><StatCard label="Tests after Holm correction" value={whole(data.promotion.test_count)} note="Family-wise multiple-testing correction applied" icon={BarChart3} color="lavender" /></div>
            <section className="panel category-panel analysis-panel"><div className="panel-heading"><div><h2>Results by analysis area</h2><p>Household- and basket-level comparisons with Holm correction</p></div></div><div className="findings-table"><div className="findings-head"><span>ANALYSIS AREA</span><span>RESULT</span><span>CONCLUSION</span></div>{promoFindings.map(item => { const status = item.Statistical_Result || ''; const significant = status.toLowerCase().includes('significant after'); return <div className="finding-row" key={item.Area}><b>{item.Area}</b><span>{item.Finding}</span><em className={significant ? 'significant-pill' : 'neutral-pill'}>{status === 'Descriptive' ? 'Descriptive' : significant ? 'Significant' : 'Not significant'}</em></div> })}</div><p className="panel-footnote">Promotion metadata indicates a recorded display or mailer match. Missing metadata is treated as “not recorded.”</p></section>
            <div className="lower-grid"><section className="panel category-panel"><div className="panel-heading"><div><h2>Basket sales by recorded promotion</h2><p>Mean basket sales</p></div></div><div className="coverage-list"><div><span>Metadata recorded</span><b>{money(data.promotion.mean_sales_with_metadata)}</b></div><div><span>No metadata recorded</span><b>{money(data.promotion.mean_sales_without_metadata)}</b></div></div></section><section className="panel category-panel"><div className="panel-heading"><div><h2>Basket discount percentage</h2><p>Mean discount percent by recorded promotion</p></div></div><div className="coverage-list"><div><span>Metadata recorded</span><b>{percent(data.promotion.mean_discount_with_metadata)}</b></div><div><span>No metadata recorded</span><b>{percent(data.promotion.mean_discount_without_metadata)}</b></div></div></section></div>
          </>}

          {page === 'Household clusters' && <>
            <PageHeader eyebrow="HOUSEHOLD SEGMENTATION" title="Household purchase patterns" description="Two groups from a PCA + Gaussian mixture model, described by their observed basket activity." action={action} />
            <div className="stat-grid"><StatCard label="Households clustered" value={whole(data.clusters.households)} note="One assignment per household" icon={Users} color="sage" /><StatCard label="Selected model" value={data.clusters.model} note={`${clusters.length} fitted components`} icon={BarChart3} color="blue" />{clusters.map((cluster, index) => <StatCard key={cluster.id} label={`Cluster ${cluster.id}`} value={whole(cluster.households)} note={`${Number(cluster.share).toFixed(2)}% of assigned households`} icon={Users} color={index === 0 ? 'peach' : 'lavender'} />)}</div>
            <div className="segment-page-grid"><section className="panel segment-list-panel"><div className="panel-heading"><div><h2>Household distribution and profile</h2><p>Median values for each model group</p></div></div><div className="cluster-list">{clusters.map(cluster => <div className="cluster-row" key={cluster.id}><div className="cluster-marker" style={{ background: cluster.id === '0' ? '#477b5e' : '#dfa66d' }} /><div className="cluster-details"><b>Cluster {cluster.id} · {cluster.activity.toLowerCase()}</b><span>Median {whole(cluster.median_baskets)} baskets · {money(cluster.median_basket_value)} average basket · {percent(cluster.median_discount_percentage)} median discount</span><div className="cluster-track"><i style={{ width: `${cluster.share}%`, background: cluster.id === '0' ? '#477b5e' : '#dfa66d' }} /></div></div><div className="cluster-count"><b>{whole(cluster.households)}</b><span>{Number(cluster.share).toFixed(2)}% of households</span></div></div>)}</div></section><section className="panel segment-insight-panel cluster-method"><div className="panel-heading"><div><h2>What separates the groups?</h2><p>Median household profile</p></div></div><div className="method-copy">{clusters.length > 1 && <><p><b>Cluster {clusters[clusters.length - 1].id}</b> shops more frequently: {whole(clusters[clusters.length - 1].median_baskets)} median baskets compared with {whole(clusters[0].median_baskets)} in Cluster {clusters[0].id}.</p><p>Its median average basket value is also higher ({money(clusters[clusters.length - 1].median_basket_value)} vs {money(clusters[0].median_basket_value)}). Median discount percentages are similar ({percent(clusters[clusters.length - 1].median_discount_percentage)} vs {percent(clusters[0].median_discount_percentage)}), so the clearest difference is purchase activity.</p></>}</div></section></div>
            <div className="insight-banner"><div className="banner-icon"><CircleHelp size={18} /></div><div><b>The clusters mainly differ in shopping activity</b><span>Cluster profiles are calculated from household basket frequency, basket value, and discount behavior.</span></div></div>
          </>}

          {page === 'Basket associations' && <>
            <PageHeader eyebrow="MARKET BASKET ANALYSIS" title="Products bought together" description="Association rules from grocery baskets, cross-checked with Apriori and FP-Growth." action={action} />
            <div className="stat-grid"><StatCard label="Apriori rules" value={whole(data.basket.rule_count)} note="Rules mined from baskets" icon={ShoppingBasket} color="sage" /><StatCard label="Frequent itemsets" value={`${whole(data.basket.itemset_count)} / ${whole(data.basket.fpgrowth_itemset_count)}`} note="Apriori / FP-Growth" icon={BarChart3} color="peach" /><StatCard label="Rules with lift above 1" value={`${whole(data.basket.high_lift_count)} / ${whole(data.basket.rule_count)}`} note="Positive basket associations" icon={TrendingUp} color="blue" /><StatCard label="Highest reported lift" value={odds(data.basket.max_lift)} note="Strongest product pairing" icon={Sparkles} color="lavender" /></div>
            <section className="panel rules-panel"><div className="panel-heading"><div><h2>Top product associations</h2><p>Rules ordered by association lift</p></div><button className="more-button" aria-label="Explain association metrics" onClick={() => notify('Support is the share of baskets containing both items. Confidence is the conditional purchase rate. Lift compares observed co-purchase to chance.')}><CircleHelp size={19} /></button></div><div className="rules-table"><div className="table-head"><span>ASSOCIATION RULE</span><span>SUPPORT</span><span>CONFIDENCE</span><span>LIFT</span></div>{data.basket.rules.map((rule, index) => <div className="rule-row" key={`${rule.antecedent}-${rule.consequent}`}><div className="rule-pair"><span className={`product-chip product-${index + 1}`}>{['↗', '◉', '✳', '◈'][index % 4]}</span><div><b>{rule.antecedent.split(' (')[0]}</b><div className="pair-connector"><span /> often appears with</div><b>{rule.consequent.split(' (')[0]}</b></div></div><span className="table-value">{percent(rule.support * 100)}</span><span className="table-value">{percent(rule.confidence * 100)}</span><span className="lift-pill">{odds(rule.lift)}</span></div>)}</div><div className="rules-footnote"><CircleHelp size={16} /><span>Association describes co-purchase patterns; it does not imply one product caused another to be purchased.</span></div></section>
            <div className="insight-banner"><div className="banner-icon"><Sparkles size={18} /></div><div><b>{data.basket.algorithms_agree ? 'Apriori and FP-Growth agree on the reported rules' : 'Apriori and FP-Growth results differ'}</b><span>{whole(data.basket.matched_rule_count)} of {whole(data.basket.rule_count)} association rules matched across methods.</span></div></div>
          </>}
          <footer className="page-footer"><span>Grocery purchase behavior study</span><span className="footer-dot" /><span>Interpret findings as associations</span></footer>
        </>}
      </div>
    </main>
    {toast && <div className="toast"><span className="toast-check">✓</span>{toast}</div>}
    {mobileNav && <button className="sidebar-scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation" />}
  </div>
}

export default App
