import { useEffect, useMemo, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  Bot,
  CheckCircle2,
  ChevronRight,
  Dumbbell,
  Loader2,
  Menu,
  Pill,
  Salad,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Target,
  X,
} from 'lucide-react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000'

const fallbackSymptoms = [
  { value: 'high_fever', label: 'High Fever' },
  { value: 'cough', label: 'Cough' },
  { value: 'fatigue', label: 'Fatigue' },
  { value: 'headache', label: 'Headache' },
  { value: 'chest_pain', label: 'Chest Pain' },
  { value: 'nausea', label: 'Nausea' },
  { value: 'skin_rash', label: 'Skin Rash' },
  { value: 'joint_pain', label: 'Joint Pain' },
  { value: 'breathlessness', label: 'Breathlessness' },
  { value: 'vomiting', label: 'Vomiting' },
]

const screens = [
  { id: 'overview', label: 'Overview' },
  { id: 'diagnosis', label: 'Diagnosis' },
  { id: 'chatbot', label: 'Chatbot' },
  { id: 'results', label: 'Results' },
  { id: 'plan', label: 'Plan' },
]

const initialChatMessages = [
  {
    role: 'assistant',
    content:
      'Hi, I can answer questions using the medical documents connected to this app. Tell me what you want to understand, and include urgent symptoms, age, medicines, allergies, or pregnancy status if relevant.',
    sources: [],
  },
]

const defaultSuggestedQuestions = [
  'What symptoms should I monitor and when should I seek care?',
  'Can you explain this condition in simple words?',
  'What warning signs are mentioned in the documents?',
]

const platformStats = [
  { label: 'tracked symptoms', value: '132' },
  { label: 'condition types', value: '41' },
  { label: 'chat guidance', value: '1:1' },
]

const patientFeatures = [
  {
    icon: Activity,
    title: 'Symptom-based diagnosis',
    body: 'Select what you are feeling and receive a ranked, probability-backed condition.',
  },
  {
    icon: ShieldCheck,
    title: 'Personal care guidance',
    body: 'Precautions, diet ideas, movement, and medication in one calm view.',
  },
  {
    icon: Bot,
    title: 'Medical chat assistant',
    body: 'Ask follow-up questions and get simple, patient-friendly answers.',
  },
]

const journeySteps = [
  { title: 'Share your symptoms', body: "Tell us what you're feeling from a guided list." },
  { title: 'Review likely conditions', body: 'See a clear, ranked explanation with confidence.' },
  { title: 'Explore your care plan', body: 'Medication, diet, exercise, and precautions together.' },
  { title: 'Ask a follow-up', body: 'Get calm, document-grounded answers anytime.' },
]

function App() {
  const [activeScreen, setActiveScreen] = useState('overview')
  const [menuOpen, setMenuOpen] = useState(false)
  const [symptoms, setSymptoms] = useState(fallbackSymptoms)
  const [selectedSymptoms, setSelectedSymptoms] = useState(['high_fever', 'cough', 'fatigue'])
  const [searchTerm, setSearchTerm] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiNotice, setApiNotice] = useState('')
  const [chatMessages, setChatMessages] = useState(initialChatMessages)
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [chatError, setChatError] = useState('')
  const [suggestedQuestions, setSuggestedQuestions] = useState(defaultSuggestedQuestions)

  useEffect(() => {
    let ignore = false

    async function loadSymptoms() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/symptoms`)
        if (!response.ok) {
          throw new Error('Symptoms endpoint is unavailable.')
        }
        const data = await response.json()
        if (!ignore && Array.isArray(data.symptoms) && data.symptoms.length > 0) {
          setSymptoms(data.symptoms)
          setApiNotice('')
        }
      } catch {
        if (!ignore) {
          setApiNotice('Using built-in symptom examples until the Flask API is running.')
        }
      }
    }

    loadSymptoms()
    return () => {
      ignore = true
    }
  }, [])

  const selectedLabels = useMemo(
    () =>
      selectedSymptoms.map(
        (symptom) => symptoms.find((option) => option.value === symptom)?.label || prettify(symptom),
      ),
    [selectedSymptoms, symptoms],
  )

  const filteredSymptoms = useMemo(() => {
    const uniqueSymptoms = Array.from(
      new Map(symptoms.map((symptom) => [symptom.value, symptom])).values(),
    )
    const query = searchTerm.trim().toLowerCase()
    if (!query) {
      return uniqueSymptoms
    }
    return uniqueSymptoms
      .filter((symptom) => `${symptom.label} ${symptom.value}`.toLowerCase().includes(query))
  }, [searchTerm, symptoms])

  const toggleSymptom = (symptom) => {
    setSelectedSymptoms((current) =>
      current.includes(symptom)
        ? current.filter((item) => item !== symptom)
        : [...current, symptom],
    )
  }

  const clearAllSymptoms = () => {
    setSelectedSymptoms([])
    setSearchTerm('')
    setResult(null)
    setError('')
  }

  const runRecommendation = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE_URL}/api/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: selectedSymptoms }),
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Recommendation failed.')
      }

      setResult(data)
      setActiveScreen('results')
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  const updateLastAssistantMessage = (updater) => {
    setChatMessages((current) => {
      const updated = [...current]
      const lastIndex = updated.length - 1
      updated[lastIndex] = updater(updated[lastIndex])
      return updated
    })
  }

  const askChatbot = async (questionOverride) => {
    const question = (questionOverride ?? chatInput).trim()
    if (!question || chatLoading) {
      return
    }

    setChatInput('')
    setChatError('')
    setChatLoading(true)
    setChatMessages((current) => [
      ...current,
      { role: 'user', content: question, sources: [] },
      { role: 'assistant', content: '', sources: [], streaming: true },
    ])

    try {
      const response = await fetch(`${API_BASE_URL}/api/chatbot/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })

      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}))
        throw new Error(data.error || 'Medical chatbot is unavailable.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let answer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })

        let newlineIndex = buffer.indexOf('\n')
        while (newlineIndex !== -1) {
          const line = buffer.slice(0, newlineIndex).trim()
          buffer = buffer.slice(newlineIndex + 1)
          newlineIndex = buffer.indexOf('\n')
          if (!line) {
            continue
          }

          const event = JSON.parse(line)
          if (event.type === 'token') {
            answer += event.content
            const nextAnswer = answer
            updateLastAssistantMessage((message) => ({ ...message, content: nextAnswer }))
          } else if (event.type === 'done') {
            const sources = Array.isArray(event.sources) ? event.sources : []
            updateLastAssistantMessage((message) => ({ ...message, sources, streaming: false }))
          } else if (event.type === 'suggestions') {
            const questions = Array.isArray(event.questions) ? event.questions.filter(Boolean) : []
            if (questions.length > 0) {
              setSuggestedQuestions(questions)
            }
          } else if (event.type === 'error') {
            throw new Error(event.message || 'Medical chatbot is unavailable.')
          }
        }
      }
    } catch (requestError) {
      const message = requestError.message
      setChatError(message)
      updateLastAssistantMessage((current) =>
        current.role === 'assistant' && !current.content
          ? { role: 'assistant', content: message, sources: [], tone: 'error', streaming: false }
          : { ...current, streaming: false },
      )
    } finally {
      setChatLoading(false)
    }
  }

  const goToScreen = (screen) => {
    setActiveScreen(screen)
    setMenuOpen(false)
  }

  return (
    <main className="min-h-screen bg-sage-50 text-sage-950">
      <header className="sticky top-0 z-50 border-b border-sage-300 bg-sage-50/90 backdrop-blur">
        <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4 lg:px-10">
          <button type="button" onClick={() => goToScreen('overview')} className="flex items-center gap-2.5 text-left">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sage-700">
              <span className="h-2.5 w-2.5 rounded-full bg-sage-50" />
            </span>
            <span className="font-serif text-xl font-semibold tracking-tight text-sage-950">Vitalis</span>
          </button>

          <ScreenNav activeScreen={activeScreen} onSelect={goToScreen} />

          <div className="hidden items-center gap-3 lg:flex">
            <button
              type="button"
              onClick={() => goToScreen('diagnosis')}
              className="text-sm font-semibold text-sage-950 underline decoration-sage-950/40 decoration-1 underline-offset-4 transition hover:text-sage-700 hover:decoration-sage-700"
            >
              Start diagnosis
            </button>
            <button
              type="button"
              onClick={() => goToScreen('chatbot')}
              className="rounded-full bg-sage-700 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-sage-800"
            >
              Talk to assistant
            </button>
          </div>

          <button
            type="button"
            className="rounded-xl border border-sage-300 bg-white p-2.5 text-sage-800 lg:hidden"
            onClick={() => setMenuOpen((value) => !value)}
            aria-label="Toggle menu"
          >
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </nav>

        {menuOpen && (
          <div className="border-t border-sage-200 bg-white px-5 py-4 lg:hidden">
            <div className="grid gap-1 text-sm font-semibold text-sage-800">
              {screens.map((screen) => (
                <button
                  key={screen.id}
                  type="button"
                  onClick={() => goToScreen(screen.id)}
                  className="rounded-lg px-3 py-2 text-left hover:bg-sage-50 hover:text-sage-700"
                >
                  {screen.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </header>

      {activeScreen === 'overview' && (
        <OverviewScreen
          selectedLabels={selectedLabels}
          onStart={() => goToScreen('diagnosis')}
          onRun={runRecommendation}
          onChat={() => goToScreen('chatbot')}
          loading={loading}
        />
      )}

      {activeScreen === 'diagnosis' && (
        <DiagnosisScreen
          apiNotice={apiNotice}
          error={error}
          filteredSymptoms={filteredSymptoms}
          loading={loading}
          onRun={runRecommendation}
          onSearch={setSearchTerm}
          onClearAll={clearAllSymptoms}
          searchTerm={searchTerm}
          selectedSymptoms={selectedSymptoms}
          symptoms={symptoms}
          toggleSymptom={toggleSymptom}
        />
      )}

      {activeScreen === 'chatbot' && (
        <ChatbotScreen
          chatError={chatError}
          input={chatInput}
          loading={chatLoading}
          messages={chatMessages}
          onAsk={askChatbot}
          onInput={setChatInput}
          suggestions={suggestedQuestions}
        />
      )}

      {activeScreen === 'results' && (
        <ResultsScreen
          result={result}
          selectedLabels={selectedLabels}
          onBack={() => goToScreen('diagnosis')}
          onPlan={() => goToScreen('plan')}
        />
      )}

      {activeScreen === 'plan' && (
        <PlanScreen result={result} onBack={() => goToScreen('results')} onStart={() => goToScreen('diagnosis')} onChat={() => goToScreen('chatbot')} />
      )}
    </main>
  )
}

function ScreenNav({ activeScreen, onSelect }) {
  return (
    <div className="hidden items-center gap-8 lg:flex">
      {screens.map((screen) => (
        <button
          key={screen.id}
          type="button"
          onClick={() => onSelect(screen.id)}
          className={`nav-link whitespace-nowrap border-b-2 pb-2 pt-1 text-[13.5px] font-semibold transition ${
            activeScreen === screen.id ? 'active border-sage-700 text-sage-950' : 'border-transparent text-sage-500'
          }`}
        >
          {screen.label}
        </button>
      ))}
    </div>
  )
}

function OverviewScreen({ selectedLabels, onStart, onRun, onChat, loading }) {
  return (
    <div>
      <section className="mx-auto max-w-7xl px-5 py-14 lg:px-10 lg:py-20">
        <div className="grid items-center gap-14 lg:grid-cols-[1.05fr_0.95fr]">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="mb-5 inline-flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-sage-500">
              <Sparkles className="h-4 w-4 text-sage-700" />
              A quieter way to check in
            </div>
            <h1 className="max-w-2xl font-serif text-4xl font-semibold leading-[1.18] tracking-tight text-sage-950 sm:text-5xl">
              Listen to your body. We&rsquo;ll help you make sense of it.
            </h1>
            <p className="mt-5 max-w-lg text-[15px] leading-relaxed text-sage-600">
              Vitalis brings symptom insight, medication guidance, nutrition, and movement together in one calm space
              — with a document-grounded assistant on hand for anything unclear.
            </p>

            <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center">
              <button
                type="button"
                onClick={onStart}
                className="group inline-flex items-center justify-center rounded-lg bg-sage-700 px-7 py-3.5 text-[13.5px] font-semibold text-white shadow-sm transition hover:bg-sage-800"
              >
                Begin symptom check
                <ChevronRight className="ml-1.5 h-4 w-4 transition group-hover:translate-x-0.5" />
              </button>
              <button
                type="button"
                onClick={onChat}
                className="inline-flex items-center justify-center border-b border-sage-950 pb-0.5 text-[13.5px] font-semibold text-sage-950 transition hover:border-sage-700 hover:text-sage-700"
              >
                Ask the assistant
                <ArrowRight className="ml-1.5 h-4 w-4" />
              </button>
            </div>

            <div className="mt-12 flex max-w-lg gap-12">
              {platformStats.map((stat) => (
                <div key={stat.label}>
                  <p className="font-serif text-[28px] font-semibold text-sage-950">{stat.value}</p>
                  <p className="mt-1 text-[11.5px] text-sage-500">{stat.label}</p>
                </div>
              ))}
            </div>
          </motion.div>

          <HeroVisual selectedLabels={selectedLabels} loading={loading} onRun={onRun} />
        </div>
      </section>

      <section className="border-t border-sage-300">
        <div className="mx-auto max-w-7xl px-5 py-14 lg:px-10">
          <div className="grid gap-6 md:grid-cols-4">
            {journeySteps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.06 }}
                className="flex flex-col items-start gap-3"
              >
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-sage-700 font-serif text-sm font-semibold text-white">
                  {index + 1}
                </span>
                <p className="text-[13.5px] font-bold text-sage-950">{step.title}</p>
                <p className="text-[12.5px] leading-relaxed text-sage-500">{step.body}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-14 lg:px-10">
        <div className="grid gap-5 md:grid-cols-3">
          {patientFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 + index * 0.06 }}
              className="rounded-2xl border border-sage-200 bg-white p-6"
            >
              <span className="flex h-11 w-11 items-center justify-center rounded-xl bg-sage-150">
                <feature.icon className="h-5 w-5 text-sage-700" />
              </span>
              <h2 className="mt-4 font-serif text-[17px] font-semibold text-sage-950">{feature.title}</h2>
              <p className="mt-2 text-[13px] leading-relaxed text-sage-600">{feature.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      <section className="border-t border-sage-300 px-5 py-14">
        <p className="mx-auto max-w-2xl text-center font-serif text-[22px] italic leading-relaxed text-sage-900">
          &ldquo;Not just a diagnosis — a clear next step: what to eat, how to rest, and when to actually see a
          doctor.&rdquo;
        </p>
        <p className="mt-4 text-center text-xs font-semibold text-sage-500">— How Vitalis thinks about care</p>
      </section>

      <div className="bg-sage-100 px-5 py-5">
        <p className="mx-auto max-w-7xl text-xs leading-relaxed text-sage-600">
          Vitalis provides educational guidance only. It is not a medical device and does not replace professional
          diagnosis, treatment, or emergency care.
        </p>
      </div>
    </div>
  )
}

function HeroVisual({ selectedLabels, loading, onRun }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.08 }}
      className="relative"
    >
      <div
        className="relative flex aspect-[4/5] items-center justify-center overflow-hidden rounded-[20px] shadow-[0_30px_70px_-30px_rgba(30,45,25,0.28)]"
        style={{ background: 'radial-gradient(circle at 50% 42%, #eef3e8 0%, #dbe6cf 55%, #cddabf 100%)' }}
      >
        <span
          className="absolute h-56 w-56 rounded-full"
          style={{ background: 'rgba(92,122,82,0.3)', animation: 'vitalisPulse 3.6s ease-out infinite' }}
        />
        <span
          className="absolute h-56 w-56 rounded-full"
          style={{ background: 'rgba(92,122,82,0.3)', animation: 'vitalisPulse 3.6s ease-out infinite 1.2s' }}
        />
        <span
          className="absolute h-56 w-56 rounded-full"
          style={{ background: 'rgba(92,122,82,0.3)', animation: 'vitalisPulse 3.6s ease-out infinite 2.4s' }}
        />

        <span
          className="absolute left-[22%] top-[28%] h-2 w-2 rounded-full bg-white/80"
          style={{ animation: 'vitalisFloat 4.5s ease-in-out infinite' }}
        />
        <span
          className="absolute left-[70%] top-[64%] h-1.5 w-1.5 rounded-full bg-white/70"
          style={{ animation: 'vitalisFloat 5.2s ease-in-out infinite 0.8s' }}
        />
        <span
          className="absolute left-[68%] top-[22%] h-1.5 w-1.5 rounded-full bg-white/60"
          style={{ animation: 'vitalisFloat 3.8s ease-in-out infinite 1.6s' }}
        />

        <div
          className="relative flex h-26 w-26 items-center justify-center rounded-full bg-sage-700 shadow-[0_18px_40px_-14px_rgba(60,80,50,0.5)]"
          style={{ animation: 'vitalisBreathe 4s ease-in-out infinite' }}
        >
          {loading ? (
            <Loader2 className="h-8 w-8 animate-spin text-sage-50" />
          ) : (
            <Stethoscope className="h-9 w-9 text-sage-50" />
          )}
        </div>
      </div>

      <div className="absolute -bottom-2 -left-6 w-56 rounded-2xl bg-white p-4 shadow-[0_20px_40px_-16px_rgba(30,45,25,0.3)] sm:-left-9 sm:bottom-8">
        <div className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sage-150">
            <Target className="h-3.5 w-3.5 text-sage-700" />
          </span>
          <div>
            <p className="text-xs font-bold text-sage-950">Your symptom summary</p>
            <p className="text-[11px] text-sage-500">{selectedLabels.length} selected</p>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {selectedLabels.slice(0, 4).map((symptom) => (
            <span key={symptom} className="rounded-full bg-sage-150 px-2.5 py-1 text-[10.5px] font-semibold text-sage-900">
              {symptom}
            </span>
          ))}
        </div>
        <button
          type="button"
          onClick={onRun}
          disabled={loading}
          className="mt-3 inline-flex w-full items-center justify-center rounded-lg bg-sage-950 px-3 py-2 text-[11.5px] font-semibold text-white transition hover:bg-sage-900 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? 'Analyzing…' : 'Try sample check'}
        </button>
      </div>
    </motion.div>
  )
}

function DiagnosisScreen({
  apiNotice,
  error,
  filteredSymptoms,
  loading,
  onClearAll,
  onRun,
  onSearch,
  searchTerm,
  selectedSymptoms,
  symptoms,
  toggleSymptom,
}) {
  return (
    <section className="mx-auto max-w-7xl px-5 py-10 lg:px-10 lg:py-14">
      <div className="grid gap-8 lg:grid-cols-[0.85fr_1.15fr]">
        <div className="self-start rounded-2xl border border-sage-200 bg-white p-6">
          <p className="text-xs font-bold uppercase tracking-[0.1em] text-sage-500">Symptom intake</p>
          <h1 className="mt-3 font-serif text-[23px] font-semibold tracking-tight text-sage-950">
            What are you noticing?
          </h1>
          <p className="mt-3 text-[13px] leading-relaxed text-sage-600">
            Select every symptom that applies. You can adjust this anytime before generating guidance.
          </p>

          <div className="mt-5 rounded-xl bg-sage-50 px-4 py-3.5">
            <p className="text-[12.5px] text-sage-600">Selected</p>
            <p className="mt-1 font-serif text-[26px] font-semibold text-sage-950">{selectedSymptoms.length}</p>
          </div>

          {apiNotice && <Notice tone="info" message={apiNotice} />}
          {error && <Notice tone="danger" message={error} />}

          <div className="mt-4 flex flex-wrap gap-2">
            {selectedSymptoms.map((symptom) => (
              <button
                key={symptom}
                type="button"
                onClick={() => toggleSymptom(symptom)}
                className="rounded-full bg-sage-150 px-3 py-1.5 text-xs font-semibold text-sage-900 transition hover:bg-sage-200"
              >
                {symptoms.find((option) => option.value === symptom)?.label || prettify(symptom)} ×
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={onRun}
            disabled={loading || selectedSymptoms.length === 0}
            className="mt-6 inline-flex w-full items-center justify-center rounded-lg bg-sage-700 px-6 py-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-sage-800 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Generating my guidance
              </>
            ) : (
              <>
                Generate my guidance
                <ArrowRight className="ml-2 h-5 w-5" />
              </>
            )}
          </button>
        </div>

        <div>
          {searchTerm && (
            <p className="mb-3 text-[12.5px] font-medium text-sage-500">
              Showing {filteredSymptoms.length} matching symptoms
            </p>
          )}
          <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-2 rounded-xl border border-sage-300 bg-sage-50 px-4 py-3">
              <Search className="h-4.5 w-4.5 text-sage-500" />
              <input
                value={searchTerm}
                onChange={(event) => onSearch(event.target.value)}
                placeholder="Search symptoms..."
                className="w-full bg-transparent text-sm font-medium text-sage-900 outline-none placeholder:text-sage-500 sm:w-64"
              />
            </div>
            <button
              type="button"
              onClick={onClearAll}
              disabled={selectedSymptoms.length === 0 && !searchTerm}
              className="rounded-xl border border-sage-300 bg-white px-4 py-3 text-sm font-semibold text-sage-800 transition hover:border-rose-200 hover:text-rose-600 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Clear all
            </button>
          </div>

          <div className="grid max-h-[620px] gap-3 overflow-y-auto pr-1 sm:grid-cols-3">
            {filteredSymptoms.map((symptom) => {
              const active = selectedSymptoms.includes(symptom.value)
              return (
                <button
                  key={symptom.value}
                  type="button"
                  onClick={() => toggleSymptom(symptom.value)}
                  className={`flex items-center justify-between rounded-xl border px-4 py-4 text-left text-[13.5px] font-semibold transition ${
                    active ? 'border-sage-700 bg-sage-150 text-sage-900' : 'border-sage-200 bg-white text-sage-950 hover:border-sage-400'
                  }`}
                >
                  <span>{symptom.label}</span>
                  <span
                    className={`h-4 w-4 shrink-0 rounded-full border-[1.5px] ${
                      active ? 'border-sage-700 bg-sage-700' : 'border-sage-400 bg-transparent'
                    }`}
                  />
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </section>
  )
}

function ChatbotScreen({ chatError, input, loading, messages, onAsk, onInput, suggestions }) {
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  return (
    <section className="mx-auto max-w-7xl px-5 py-10 lg:px-10 lg:py-14">
      <div className="grid gap-6 lg:grid-cols-[0.62fr_1.38fr]">
        <div className="flex flex-col gap-4 self-start">
          <div className="rounded-2xl border border-sage-200 bg-white p-6">
            <span className="flex h-10 w-10 items-center justify-center rounded-full bg-sage-150">
              <Bot className="h-4.5 w-4.5 text-sage-700" />
            </span>
            <p className="mt-4 text-xs font-bold uppercase tracking-[0.1em] text-sage-500">Medical RAG assistant</p>
            <h1 className="mt-2 font-serif text-[22px] font-semibold tracking-tight text-sage-950">
              A question, answered calmly.
            </h1>
            <p className="mt-2.5 text-[13px] leading-relaxed text-sage-600">
              This assistant answers from connected medical documents. It explains — it does not diagnose.
            </p>
          </div>

          <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
            <p className="text-[12.5px] font-bold text-amber-800">Seek urgent care for:</p>
            <p className="mt-1.5 text-xs leading-relaxed text-amber-800">
              Chest pain, breathing trouble, stroke-like symptoms, severe bleeding, or sudden worsening.
            </p>
          </div>

          {chatError && <Notice tone="danger" message={chatError} />}

          <div className="rounded-2xl border border-sage-200 bg-white p-5">
            <p className="text-xs font-bold uppercase tracking-[0.08em] text-sage-500">Try asking</p>
            <div className="mt-3 flex flex-col gap-2">
              {suggestions.map((question) => (
                <button
                  key={question}
                  type="button"
                  onClick={() => onAsk(question)}
                  disabled={loading}
                  className="rounded-xl bg-sage-50 p-3 text-left text-[12.5px] leading-relaxed text-sage-900 transition hover:bg-sage-150 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex flex-col rounded-2xl border border-sage-200 bg-white p-6">
          <div className="flex items-center justify-between gap-4 border-b border-sage-200 pb-4">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.1em] text-sage-500">Patient chat</p>
              <h2 className="mt-1 font-serif text-lg font-semibold text-sage-950">Document-grounded answers</h2>
            </div>
            {loading && <Loader2 className="h-5 w-5 animate-spin text-sage-700" />}
          </div>

          <div className="mt-5 flex max-h-[520px] flex-col gap-5 overflow-y-auto pr-1">
            {messages.map((message, index) => (
              <ChatMessage key={`${message.role}-${index}`} message={message} />
            ))}
            <div ref={chatEndRef} />
          </div>

          <div className="mt-5 rounded-xl border border-sage-300 bg-sage-50 p-3">
            <textarea
              value={input}
              onChange={(event) => onInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  onAsk()
                }
              }}
              placeholder="Ask a question about symptoms, medication, or care..."
              rows={3}
              className="min-h-24 w-full resize-none bg-transparent p-2 text-sm font-medium leading-6 text-sage-900 outline-none placeholder:text-sage-500"
            />
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs text-sage-500">Press Enter to send, Shift + Enter for a new line.</p>
              <button
                type="button"
                onClick={() => onAsk()}
                disabled={loading || input.trim().length === 0}
                className="inline-flex items-center rounded-lg bg-sage-700 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-sage-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? 'Thinking...' : 'Send'}
                {loading ? <Loader2 className="ml-2 h-4 w-4 animate-spin" /> : <Send className="ml-2 h-4 w-4" />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className="max-w-[80%]">
        <p
          className={`mb-1.5 text-[10.5px] font-bold uppercase tracking-[0.06em] ${
            isUser ? 'text-right text-sage-500' : 'text-sage-700'
          }`}
        >
          {isUser ? 'You' : 'Vitalis assistant'}
        </p>
        <div
          className={`rounded-2xl p-4 text-[13.5px] leading-relaxed ${
            isUser
              ? 'bg-sage-950 text-white'
              : message.tone === 'error'
                ? 'border border-rose-200 bg-rose-50 text-rose-900'
                : 'bg-sage-50 text-sage-950'
          }`}
        >
          {message.streaming && !message.content ? (
            <div className="flex items-center gap-1.5 py-1">
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-500 [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-500 [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-sage-500" />
            </div>
          ) : (
            <p className="whitespace-pre-line">
              {message.content}
              {message.streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-sage-500 align-middle" />}
            </p>
          )}

          {!isUser && message.sources?.length > 0 && (
            <div className="mt-4 grid gap-2">
              <p className="text-[10.5px] font-bold uppercase tracking-[0.1em] text-sage-500">Sources</p>
              {message.sources.slice(0, 4).map((source, index) => (
                <div key={`${source.source}-${index}`} className="rounded-lg bg-white p-2.5 text-xs font-medium leading-5 text-sage-600">
                  <span className="font-semibold text-sage-950">{source.source || 'Unknown source'}</span>
                  {source.page != null && <span> · page {Number(source.page) + 1}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ResultsScreen({ result, selectedLabels, onBack, onPlan }) {
  if (!result) {
    return <EmptyState title="No recommendation yet" body="Run a diagnosis first to view results." action="Start diagnosis" onAction={onBack} />
  }

  const confidence = Math.max(0, Math.min(Number(result.confidence_percent) || 0, 100))

  return (
    <section className="mx-auto max-w-7xl px-5 py-10 lg:px-10 lg:py-14">
      <div className="grid gap-6 rounded-2xl border border-sage-200 bg-white p-8 lg:grid-cols-[auto_1fr] lg:items-center">
        <div
          className="flex h-32 w-32 shrink-0 items-center justify-center rounded-full"
          style={{ background: `conic-gradient(#5c7a52 0% ${confidence}%, #e5ecdd ${confidence}% 100%)` }}
        >
          <div className="flex h-24 w-24 flex-col items-center justify-center rounded-full bg-white">
            <p className="font-serif text-2xl font-semibold text-sage-950">{formatPercent(result.confidence_percent)}</p>
            <p className="mt-0.5 text-[10px] font-bold uppercase tracking-wide text-sage-500">confidence</p>
          </div>
        </div>
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.1em] text-sage-500">Most likely condition</p>
          <h1 className="mt-2 font-serif text-[28px] font-semibold tracking-tight text-sage-950">{result.disease}</h1>
          <p className="mt-2.5 max-w-2xl text-[13.5px] leading-relaxed text-sage-600">
            {result.description || 'No description is available for this condition.'}
          </p>
          <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-sage-500">
            Reliability: <span className="text-sage-900">{result.reliability || 'Unknown'}</span>
          </p>
        </div>
      </div>

      <div className="mt-6 rounded-2xl border border-sage-200 bg-white p-6">
        <p className="mb-3 text-xs font-bold uppercase tracking-[0.1em] text-sage-500">Selected symptoms</p>
        <div className="flex flex-wrap gap-2">
          {selectedLabels.map((symptom) => (
            <span key={symptom} className="rounded-full bg-sage-150 px-3 py-1.5 text-xs font-semibold text-sage-900">
              {symptom}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-2">
        <Panel title="Other possibilities" icon={Brain}>
          <div className="grid gap-4">
            {(result.top_predictions || []).map((prediction) => (
              <ProbabilityBar
                key={prediction.disease}
                label={prediction.disease}
                value={prediction.probability_percent}
              />
            ))}
          </div>
        </Panel>

        {(result.warnings?.length > 0 || result.safety_notes?.length > 0) && (
          <Panel title="Safety review" icon={AlertTriangle} tone="amber">
            <div className="grid gap-3">
              {[...(result.safety_notes || []), ...(result.warnings || [])].map((item) => (
                <div key={item} className="rounded-xl border border-amber-200 bg-amber-50 p-3.5 text-[13px] leading-relaxed text-amber-800">
                  {item}
                </div>
              ))}
            </div>
          </Panel>
        )}
      </div>

      <div className="mt-6 grid gap-5 md:grid-cols-2">
        <ListCard title="Medications" icon={Pill} items={result.medications} empty="No medication data available." />
        <ListCard title="Diet" icon={Salad} items={result.diets} empty="No diet data available." />
        <ListCard title="Exercises" icon={Dumbbell} items={result.exercises?.slice(0, 8)} empty="No exercise data available." />
        <ListCard title="Precautions" icon={ShieldCheck} items={result.precautions} empty="No precautions available." />
      </div>

      <div className="mt-6 flex items-center justify-between gap-4 rounded-2xl bg-sage-100 p-5">
        <p className="text-xs leading-relaxed text-sage-600">
          This guidance is educational only and does not replace professional medical diagnosis, treatment, or
          emergency care.
        </p>
        <button
          type="button"
          onClick={onPlan}
          className="inline-flex shrink-0 items-center rounded-full bg-sage-950 px-5 py-2.5 text-xs font-semibold text-white transition hover:bg-sage-900"
        >
          Open care plan
          <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
        </button>
      </div>
    </section>
  )
}

function PlanScreen({ result, onBack, onStart, onChat }) {
  if (!result) {
    return <EmptyState title="No care plan yet" body="Generate recommendations first to build a care plan." action="Start diagnosis" onAction={onStart} />
  }

  return (
    <section className="mx-auto max-w-7xl px-5 py-10 lg:px-10 lg:py-14">
      <div className="mb-6 flex flex-col justify-between gap-4 rounded-2xl border border-sage-200 bg-white p-6 sm:flex-row sm:items-end">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.1em] text-sage-500">Patient guidance</p>
          <h1 className="mt-2 font-serif text-[26px] font-semibold tracking-tight text-sage-950">
            Care plan for {result.disease}
          </h1>
          <p className="mt-2.5 max-w-2xl text-[13.5px] leading-relaxed text-sage-600">
            A practical plan assembled from the prediction, precautions, nutrition suggestions, activity guidance,
            and safety notes.
          </p>
        </div>
        <button
          type="button"
          onClick={onBack}
          className="shrink-0 rounded-full border border-sage-300 bg-white px-5 py-2.5 text-sm font-semibold text-sage-800 transition hover:border-sage-700 hover:text-sage-700"
        >
          Back to results
        </button>
      </div>

      <div className="grid gap-5 lg:grid-cols-4">
        {(result.care_plan || []).map((section, index) => (
          <motion.div
            key={section.title}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="rounded-2xl border border-sage-200 bg-white p-5"
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-sage-150 text-sm font-bold text-sage-700">
              {index + 1}
            </span>
            <h2 className="mt-4 font-serif text-[15px] font-semibold text-sage-950">{section.title}</h2>
            <div className="mt-3 grid gap-2.5">
              {(section.items || []).map((item) => (
                <div key={item} className="flex gap-2.5 rounded-xl bg-sage-50 p-3 text-[13px] leading-relaxed text-sage-700">
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-sage-700" />
                  <span>{item}</span>
                </div>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-6 flex flex-col items-start gap-4 rounded-2xl border border-amber-200 bg-amber-50 p-5 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-[13px] leading-relaxed text-amber-800">
          This project provides educational support only. It is not a medical device and does not replace
          professional medical diagnosis, treatment, or emergency care.
        </p>
        <button
          type="button"
          onClick={onChat}
          className="inline-flex shrink-0 items-center rounded-full bg-sage-950 px-5 py-2.5 text-xs font-semibold text-white transition hover:bg-sage-900"
        >
          Ask about this
        </button>
      </div>
    </section>
  )
}

function Notice({ tone, message }) {
  const styles = tone === 'danger' ? 'border-rose-200 bg-rose-50 text-rose-800' : 'border-sage-300 bg-sage-50 text-sage-800'
  return <div className={`mt-4 rounded-xl border p-3.5 text-[13px] leading-relaxed ${styles}`}>{message}</div>
}

function Panel({ title, icon: Icon, tone = 'sage', children }) {
  const toneStyles = tone === 'amber' ? 'bg-amber-50 text-amber-800' : 'bg-sage-150 text-sage-700'
  return (
    <div className="rounded-2xl border border-sage-200 bg-white p-6">
      <div className="mb-5 flex items-center gap-3">
        <span className={`flex h-9 w-9 items-center justify-center rounded-xl ${toneStyles}`}>
          <Icon className="h-4.5 w-4.5" />
        </span>
        <h2 className="font-serif text-[15px] font-semibold text-sage-950">{title}</h2>
      </div>
      {children}
    </div>
  )
}

function ListCard({ title, icon, items = [], empty }) {
  const Icon = icon
  return (
    <Panel title={title} icon={Icon}>
      <div className="grid gap-2.5">
        {items.length > 0 ? (
          items.map((item) => (
            <div key={item} className="rounded-xl bg-sage-50 p-3 text-[13px] leading-relaxed text-sage-700">
              {item}
            </div>
          ))
        ) : (
          <div className="rounded-xl bg-sage-50 p-3 text-[13px] leading-relaxed text-sage-500">{empty}</div>
        )}
      </div>
    </Panel>
  )
}

function ProbabilityBar({ label, value }) {
  const safeValue = Math.max(0, Math.min(Number(value) || 0, 100))
  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-4 text-[13px] font-semibold text-sage-800">
        <span>{label}</span>
        <span className="text-sage-700">{safeValue.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-sage-100">
        <motion.div
          className="h-full rounded-full bg-sage-700"
          initial={{ width: 0 }}
          animate={{ width: `${safeValue}%` }}
          transition={{ duration: 0.6 }}
        />
      </div>
    </div>
  )
}

function EmptyState({ title, body, action, onAction }) {
  return (
    <section className="mx-auto flex min-h-[70vh] max-w-3xl items-center px-5 py-16">
      <div className="w-full rounded-2xl border border-sage-200 bg-white p-8 text-center">
        <span className="mx-auto flex h-13 w-13 items-center justify-center rounded-full bg-sage-150">
          <Activity className="h-6 w-6 text-sage-700" />
        </span>
        <h1 className="mt-5 font-serif text-2xl font-semibold text-sage-950">{title}</h1>
        <p className="mt-2.5 text-sage-600">{body}</p>
        <button
          type="button"
          onClick={onAction}
          className="mt-6 rounded-full bg-sage-700 px-6 py-3 text-sm font-semibold text-white transition hover:bg-sage-800"
        >
          {action}
        </button>
      </div>
    </section>
  )
}

function prettify(value) {
  return String(value).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function formatPercent(value) {
  return value == null ? 'N/A' : `${Number(value).toFixed(1)}%`
}

export default App
