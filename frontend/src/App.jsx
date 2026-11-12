import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, dateQuery } from "./api.js";

const NAV_ITEMS = [
  ["Dashboard", "Overview"],
  ["Patient Profile", "Patient profile"],
  ["Meal History", "Meal history"],
  ["Glucose Trends", "Glucose trends"],
  ["Lifestyle", "Lifestyle"],
  ["Analytics", "Analytics"],
  ["Meal Recommendations", "Meal recommendations"],
  ["🤖 AI Assistant", "AI assistant"],
  ["Medical Knowledge", "Medical knowledge"],
  ["Food Nutrition", "Food nutrition"],
];

function display(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return `${Number(value.toFixed(1))}${suffix}`;
  return `${value}${suffix}`;
}

function dateLabel(value) {
  if (!value) return "—";
  return new Date(`${value}T00:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function SectionTitle({ title, description, children }) {
  return (
    <div className="section-title">
      <div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {children}
    </div>
  );
}

function Metric({ label, value, note, accent = "" }) {
  return (
    <article className={`metric-card ${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </article>
  );
}

function EmptyState({ children = "No records are available for this period." }) {
  return <div className="empty-state">{children}</div>;
}

function DataTable({ rows, columns }) {
  if (!rows?.length) return <EmptyState />;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>{columns.map(([key, label]) => <th key={key}>{label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.meal_id || row.reading_id || row.lifestyle_id || row.food_id || index}>
              {columns.map(([key]) => <td key={key}>{display(row[key])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Dashboard({ data, onNavigate }) {
  const analytics = data?.analytics;
  const meals = analytics?.meal_summary || {};
  const glucose = analytics?.glucose_summary || {};
  const lifestyle = analytics?.lifestyle_summary || {};
  const profile = data?.patient || analytics?.patient_profile || {};
  const glucosePoints = glucose.daily_statistics || [];
  const mealPoints = meals.daily_nutrition_from_recorded_meals || [];

  if (!analytics) return <EmptyState>Dashboard information is not available.</EmptyState>;

  return (
    <>
      <SectionTitle
        title={`Good to see you, ${profile.patient_id || "there"}`}
        description={`A descriptive view of recorded data from ${dateLabel(analytics.start_date)} to ${dateLabel(analytics.end_date)}.`}
      >
        {profile.is_demo && <span className="demo-pill">Fictional demo profile</span>}
      </SectionTitle>
      <div className="metric-grid">
        <Metric label="Recorded meals" value={meals.recorded_meal_count ?? 0} note="Selected period" />
        <Metric label="Glucose readings" value={glucose.reading_count ?? 0} note="Recorded measurements" accent="mint" />
        <Metric label="Average recorded glucose" value={display(glucose.average_recorded_glucose, " mg/dL")} note="Not a target or interpretation" accent="blue" />
        <Metric label="Lifestyle records" value={lifestyle.record_count ?? 0} note="Selected period" accent="lavender" />
      </div>
      <div className="content-grid two">
        <article className="panel chart-panel">
          <div className="panel-heading"><h2>Recorded glucose</h2><span>Descriptive only</span></div>
          {glucosePoints.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={glucosePoints}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tickFormatter={dateLabel} />
                <YAxis />
                <Tooltip labelFormatter={dateLabel} />
                <Line type="monotone" dataKey="average_recorded_glucose" name="Daily average" stroke="#168b70" strokeWidth={3} dot />
              </LineChart>
            </ResponsiveContainer>
          ) : <EmptyState>No glucose readings in the selected period.</EmptyState>}
        </article>
        <article className="panel chart-panel">
          <div className="panel-heading"><h2>Recorded meals</h2><span>Nutrition data may be estimates</span></div>
          {mealPoints.length ? (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={mealPoints}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tickFormatter={dateLabel} />
                <YAxis />
                <Tooltip labelFormatter={dateLabel} />
                <Area type="monotone" dataKey="calories" name="Calories from recorded meals" stroke="#e29a37" fill="#fbe8c7" />
              </AreaChart>
            </ResponsiveContainer>
          ) : <EmptyState>No meals in the selected period.</EmptyState>}
        </article>
      </div>
      <article className="panel quality-panel">
        <h2>About this information</h2>
        <p>GlucoCare is an educational self-management prototype, not a clinical decision-support system. Demo records are fictional. Recorded nutrition totals may not represent a complete day.</p>
        <div className="quick-links">
          <button onClick={() => onNavigate("Analytics")}>View analytics</button>
          <button onClick={() => onNavigate("🤖 AI Assistant")}>Ask the assistant</button>
          <button onClick={() => onNavigate("Meal Recommendations")}>Browse food suggestions</button>
        </div>
      </article>
    </>
  );
}

function Profile({ patient }) {
  if (!patient) return <EmptyState>Patient profile not available.</EmptyState>;
  const fields = [
    ["Patient ID", patient.patient_id],
    ["Age", patient.age],
    ["Gender", patient.gender],
    ["Diabetes type", patient.diabetes_type],
    ["Diet preference", patient.diet_preference],
    ["Cuisine preference", patient.cuisine_preference],
    ["Activity level", patient.activity_level],
    ["Allergies / restrictions", patient.allergies || "None recorded"],
    ["Health goals", patient.health_goals],
    ["Lifestyle notes", patient.lifestyle_notes],
  ];
  return (
    <>
      <SectionTitle title="Patient profile" description="Profile details recorded for the selected demo patient." />
      <article className="panel profile-grid">
        {fields.map(([label, value]) => <div key={label}><span>{label}</span><strong>{display(value)}</strong></div>)}
      </article>
      <p className="muted-note">Profile data is fictional demo information, not a medical record.</p>
    </>
  );
}

function Analytics({ data }) {
  const meal = data?.meal_summary || {};
  const glucose = data?.glucose_summary || {};
  const lifestyle = data?.lifestyle_summary || {};
  const quality = data?.data_quality || {};
  return (
    <>
      <SectionTitle title="Patient analytics" description="Descriptive summaries; associations do not demonstrate causation." />
      <div className="metric-grid">
        <Metric label="Meal records" value={meal.recorded_meal_count ?? 0} />
        <Metric label="Glucose records" value={glucose.reading_count ?? 0} accent="mint" />
        <Metric label="Average sleep" value={display(lifestyle.average_sleep_hours, " h")} accent="blue" />
        <Metric label="Average exercise" value={display(lifestyle.average_exercise_minutes, " min")} accent="lavender" />
      </div>
      <div className="content-grid two">
        <article className="panel chart-panel">
          <h2>Glucose by day</h2>
          {(glucose.daily_statistics || []).length ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={glucose.daily_statistics}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="date" tickFormatter={dateLabel} /><YAxis /><Tooltip labelFormatter={dateLabel} />
                <Legend /><Line dataKey="minimum_recorded_glucose" name="Minimum" stroke="#74a99b" />
                <Line dataKey="maximum_recorded_glucose" name="Maximum" stroke="#d08342" />
              </LineChart>
            </ResponsiveContainer>
          ) : <EmptyState>No glucose trend for this period.</EmptyState>}
        </article>
        <article className="panel chart-panel">
          <h2>Meals by type</h2>
          {Object.keys(meal.meal_type_distribution || {}).length ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={Object.entries(meal.meal_type_distribution).map(([name, count]) => ({ name, count }))}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" /><YAxis allowDecimals={false} /><Tooltip />
                <Bar dataKey="count" fill="#168b70" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <EmptyState>No meal records for this period.</EmptyState>}
        </article>
      </div>
      <article className="panel">
        <h2>Data availability</h2>
        <p>{quality.note || "Missing dates indicate days without any meal, glucose, or lifestyle record."}</p>
        <div className="quality-grid">
          <span>Dates without records <strong>{quality.missing_date_count ?? 0}</strong></span>
          <span>Missing glucose timestamps <strong>{quality.missing_glucose_timestamps ?? 0}</strong></span>
          <span>Missing meal timestamps <strong>{quality.missing_meal_timestamps ?? 0}</strong></span>
          <span>Meal–glucose pairs <strong>{quality.meal_glucose_association_count ?? 0}</strong></span>
        </div>
        <p className="muted-note">{quality.meal_glucose_association_status || "Associations are descriptive and do not establish causality."}</p>
      </article>
    </>
  );
}

function Assistant({ patientId, startDate, endDate }) {
  const [consent, setConsent] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setMessages([]);
    setConsent(false);
  }, [patientId, startDate, endDate]);

  async function submit(event) {
    event.preventDefault();
    if (!consent || !question.trim() || pending) return;
    const userMessage = { role: "user", content: question.trim() };
    const next = [...messages, userMessage];
    setMessages(next);
    setQuestion("");
    setPending(true);
    setError("");
    try {
      const result = await api("/api/assistant", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientId,
          question: userMessage.content,
          start_date: startDate,
          end_date: endDate,
          history: messages.map(({ role, content }) => ({ role, content })),
        }),
      });
      setMessages([...next, { role: "assistant", content: result.answer, sources: result.sources || [], context: result.context_used || {} }]);
    } catch (requestError) {
      setMessages(next);
      setError(requestError.message);
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <SectionTitle title="AI assistant" description="Ask about this patient's recorded data or general diabetes education." />
      <div className="notice warning">Educational prototype only. Not for diagnosis, emergencies, or medication decisions. Do not send sensitive information.</div>
      <div className="notice consent">
        <label><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
          I consent to sending my question, relevant selected-patient context, retrieved passages, and session chat history to the configured AI provider.</label>
      </div>
      <section className="chat-panel panel">
        {messages.length === 0 && <div className="suggestion-box"><h2>Try asking</h2><div className="suggestions">
          {["Summarize my recent glucose readings", "What meals have I recorded?", "Summarize my recent lifestyle records", "What general diabetes education does the knowledge base provide?"].map((item) =>
            <button key={item} onClick={() => setQuestion(item)}>{item}</button>)}
        </div></div>}
        <div className="chat-history">
          {messages.map((message, index) => (
            <article key={`${index}-${message.role}`} className={`chat-message ${message.role}`}>
              <span>{message.role === "user" ? "You" : "GlucoCare"}</span>
              <p>{message.content}</p>
              {message.sources?.length > 0 && <details><summary>Sources ({message.sources.length})</summary>
                {message.sources.map((source, sourceIndex) => <p key={sourceIndex}>{source.title || source.source}: {source.url ? <a href={source.url} target="_blank" rel="noreferrer">{source.url}</a> : source.source}</p>)}
              </details>}
              {message.context && Object.values(message.context).some(Boolean) && <details><summary>Patient context used</summary><p>{Object.entries(message.context).filter(([, used]) => used).map(([name]) => name.replaceAll("_", " ")).join(", ")}</p></details>}
            </article>
          ))}
          {pending && <p className="muted-note">Preparing a grounded answer…</p>}
        </div>
        {error && <div className="error-banner" role="alert">{error}</div>}
        <form className="chat-form" onSubmit={submit}>
          <textarea value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about records or general diabetes education…" disabled={!consent || pending} rows="2" />
          <button className="primary-button" disabled={!consent || pending || !question.trim()}>{pending ? "Sending…" : "Send"}</button>
        </form>
        {!consent && <small className="muted-note">Consent is required before sending a question.</small>}
      </section>
    </>
  );
}

function MedicalKnowledge() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function search(event) {
    event.preventDefault();
    if (!query.trim()) return;
    setPending(true);
    setError("");
    try {
      const params = new URLSearchParams({ query: query.trim() });
      const result = await api(`/api/medical/search?${params}`);
      setResults(result.results || []);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <SectionTitle title="Medical knowledge" description="Search the local, source-attributed knowledge base." />
      <form className="search-form panel" onSubmit={search}>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. diabetes self-management education" />
        <button className="primary-button" disabled={pending}>{pending ? "Searching…" : "Search"}</button>
      </form>
      {error && <div className="error-banner" role="alert">{error}</div>}
      {results.length === 0 && !error ? <EmptyState>Search for a topic to see relevant source passages.</EmptyState> :
        results.map((item, index) => <article className="panel evidence-card" key={`${item.url}-${index}`}>
          <div className="panel-heading"><h2>{item.title || item.source}</h2><span>{item.topic}</span></div>
          <p>{item.text}</p>
          {item.url && <a href={item.url} target="_blank" rel="noreferrer">Open source ↗</a>}
        </article>)}
    </>
  );
}

function App() {
  const [patients, setPatients] = useState([]);
  const [patientId, setPatientId] = useState("");
  const [patient, setPatient] = useState(null);
  const [page, setPage] = useState("Dashboard");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const query = useMemo(() => dateQuery(startDate, endDate), [startDate, endDate]);

  useEffect(() => {
    let active = true;
    Promise.all([api("/api/health"), api("/api/patients")])
      .then(([health, patientRows]) => {
        if (!active) return;
        setApiOnline(Boolean(health));
        setPatients(patientRows);
        if (patientRows.length) setPatientId(patientRows[0].patient_id);
        else setLoading(false);
      })
      .catch((requestError) => {
        if (!active) return;
        setError(requestError.message);
        setLoading(false);
      });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!patientId) return undefined;
    let active = true;
    setLoading(true);
    setError("");
    Promise.all([
      api(`/api/patients/${encodeURIComponent(patientId)}`),
      api(`/api/patients/${encodeURIComponent(patientId)}/dashboard`),
    ])
      .then(([profile, dashboard]) => {
        if (!active) return;
        setPatient(profile);
        setData(dashboard);
        if (!startDate && dashboard.analytics?.start_date) setStartDate(dashboard.analytics.start_date);
        if (!endDate && dashboard.analytics?.end_date) setEndDate(dashboard.analytics.end_date);
        setLoading(false);
      })
      .catch((requestError) => {
        if (active) {
          setError(requestError.message);
          setLoading(false);
        }
      });
    return () => { active = false; };
  }, [patientId]);

  const loadPage = useCallback(async () => {
    if (!patientId || !startDate || !endDate) return;
    setLoading(true);
    setError("");
    const patientPath = `/api/patients/${encodeURIComponent(patientId)}`;
    try {
      let result;
      if (page === "Dashboard" || page === "Analytics") {
        result = await api(`${patientPath}/dashboard${query}`);
        setData(result);
      } else if (page === "Patient Profile") {
        result = await api(patientPath);
        setPatient(result);
      } else if (page === "Meal History" || page === "Glucose Trends" || page === "Lifestyle") {
        const resource = page === "Meal History" ? "meals" : page === "Glucose Trends" ? "glucose" : "lifestyle";
        result = await api(`${patientPath}/${resource}${query}`);
        setData(result);
      } else if (page === "Food Nutrition") {
        result = await api("/api/foods");
        setData(result);
      } else if (page === "Meal Recommendations") {
        result = await api(`${patientPath}/recommendations`);
        setData(result);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [patientId, page, query, startDate, endDate]);

  useEffect(() => { loadPage(); }, [loadPage]);

  function renderPage() {
    if (page === "Dashboard") return <Dashboard data={data} onNavigate={setPage} />;
    if (page === "Patient Profile") return <Profile patient={patient} />;
    if (page === "Meal History") return <><SectionTitle title="Meal history" description="Meals recorded for the selected period." /><DataTable rows={data} columns={[["date", "Date"], ["meal_time", "Time"], ["meal_type", "Meal"], ["food", "Food"], ["portion", "Portion"], ["calories", "Calories"], ["carbohydrates", "Carbohydrates (g)"]]} /></>;
    if (page === "Glucose Trends") return <>
      <SectionTitle title="Glucose trends" description="Recorded readings only. No target ranges or diagnosis are shown." />
      {data?.length ? <article className="panel chart-panel"><ResponsiveContainer width="100%" height={320}><LineChart data={[...data].reverse()}><CartesianGrid strokeDasharray="3 3" vertical={false} /><XAxis dataKey="date" tickFormatter={dateLabel} /><YAxis /><Tooltip /><Line dataKey="glucose_value" name="Recorded glucose (mg/dL)" stroke="#168b70" strokeWidth={3} /></LineChart></ResponsiveContainer></article> : <EmptyState>No readings for the selected period.</EmptyState>}
      <DataTable rows={data} columns={[["date", "Date"], ["time", "Time"], ["glucose_value", "Glucose (mg/dL)"], ["context", "Context"]]} />
    </>;
    if (page === "Lifestyle") return <><SectionTitle title="Lifestyle history" description="Descriptive view of recorded lifestyle entries." /><DataTable rows={data} columns={[["date", "Date"], ["sleep_hours", "Sleep (hours)"], ["exercise_minutes", "Exercise (minutes)"], ["water_intake", "Water (liters)"], ["stress_level", "Stress level"]]} /></>;
    if (page === "Analytics") return <Analytics data={data?.analytics} />;
    if (page === "Meal Recommendations") return <>
      <SectionTitle title="Meal recommendations" description="General suggestions from the food database; not medical or dietary prescriptions." />
      <div className="recommendation-grid">{data?.length ? data.map((item) => <article className="panel recommendation-card" key={item.food_name}><span className="category-pill">{item.category}</span><h2>{item.food_name}</h2><p>{item.reason}</p>{item.calories !== null && <small>{item.calories} kcal (database entry)</small>}{item.is_demo && <small className="demo-pill">Demo estimate, unverified</small>}</article>) : <EmptyState>No matching food entries are currently available.</EmptyState>}</div>
    </>;
    if (page === "🤖 AI Assistant") return <Assistant key={`${patientId}-${startDate}-${endDate}`} patientId={patientId} startDate={startDate} endDate={endDate} />;
    if (page === "Medical Knowledge") return <MedicalKnowledge />;
    if (page === "Food Nutrition") return <><SectionTitle title="Food nutrition reference" description="Nutrition values may be illustrative estimates. Always check actual ingredients and serving sizes." /><DataTable rows={data} columns={[["food_name", "Food"], ["regional_name", "Regional name"], ["category", "Category"], ["serving_size", "Serving size"], ["calories", "Calories"], ["carbohydrates", "Carbohydrates (g)"], ["protein", "Protein (g)"], ["fiber", "Fiber (g)"], ["source", "Source"]]} /></>;
    return null;
  }

  if (loading && patients.length === 0) return <main className="boot-screen"><div className="brand-mark">G</div><h1>GlucoCare</h1><p>Connecting to the application API…</p>{error && <div className="error-banner">{error}</div>}</main>;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setPage("Dashboard"); }}>
          <span className="brand-mark">G</span><span><strong>GlucoCare</strong><small>Care, made clearer</small></span>
        </a>
        <div className="side-label">WORKSPACE</div>
        <nav>{NAV_ITEMS.map(([id, label]) => <button key={id} className={page === id ? "nav-item active" : "nav-item"} onClick={() => setPage(id)}><span className="nav-dot" />{label}</button>)}</nav>
        <div className="sidebar-footer"><span className={apiOnline ? "status-dot online" : "status-dot"} />{apiOnline ? "API connected" : "API offline"}<small>Educational prototype</small></div>
      </aside>
      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">GlucoCare <span>/</span> {page.replace("🤖 ", "")}</div>
          <div className="top-controls">
            <label className="patient-picker"><span>Patient</span><select value={patientId} onChange={(event) => { setPatientId(event.target.value); setStartDate(""); setEndDate(""); }} aria-label="Select demo patient">{patients.map((item) => <option key={item.patient_id} value={item.patient_id}>{item.patient_id}</option>)}</select></label>
            {!["🤖 AI Assistant", "Medical Knowledge", "Meal Recommendations", "Food Nutrition", "Patient Profile"].includes(page) && <>
              <label className="date-picker"><span>From</span><input type="date" value={startDate} max={endDate || undefined} onChange={(event) => setStartDate(event.target.value)} /></label>
              <label className="date-picker"><span>To</span><input type="date" value={endDate} min={startDate || undefined} onChange={(event) => setEndDate(event.target.value)} /></label>
            </>}
          </div>
        </header>
        <section className="page-content">
          {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError("")}>Dismiss</button></div>}
          {loading && patients.length > 0 && <div className="loading-bar" aria-label="Loading" />}
          {renderPage()}
          <footer className="page-footer">GlucoCare is an academic educational prototype, not a clinical decision-support system. Consult a qualified healthcare professional for medical decisions.</footer>
        </section>
      </main>
    </div>
  );
}

export default App;
