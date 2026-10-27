import Image from "next/image";
import Link from "next/link";
import { Section } from "@/components/Section";
import { SITE } from "@/lib/site";

export default function HomePage() {
  const h = SITE.home;
  return (
    <>
      {/* Hero */}
      <div className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(50%_50%_at_30%_10%,rgba(139,92,246,0.18),transparent_70%),radial-gradient(40%_40%_at_80%_20%,rgba(34,211,238,0.16),transparent_70%)]" />
        <div className="container-x relative grid items-center gap-12 py-16 sm:py-24 lg:grid-cols-2">
          <div>
            <p className="eyebrow">{h.eyebrow}</p>
            <h1 className="mt-4 text-4xl font-semibold leading-tight tracking-tight text-white sm:text-5xl lg:text-6xl">
              {h.headline}
            </h1>
            <p className="mt-6 max-w-xl text-lg text-slate-400">{h.subheadline}</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href={h.primary_cta.href} className="btn-primary">{h.primary_cta.label}</Link>
              <Link href={h.secondary_cta.href} className="btn-secondary">{h.secondary_cta.label}</Link>
            </div>
            <p className="mt-6 text-xs text-slate-500">{SITE.company.status_note}</p>
          </div>
          <div className="card !p-2">
            <Image src="/images/velos-qpi-concept.jpg" alt="Velos QPI product concept: compact 630 nm live-cell imaging hardware with the SPARK reconstruction platform producing amplitude and quantitative phase images"
              width={1800} height={1012} priority className="rounded-xl" />
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="border-y border-white/10 bg-white/[0.02]">
        <div className="container-x grid grid-cols-2 gap-6 py-10 md:grid-cols-4">
          {h.stats.map((s) => (
            <div key={s.value}>
              <div className="text-3xl font-semibold text-white sm:text-4xl"><span className="gradient-text">{s.value}</span></div>
              <p className="mt-1 text-sm text-slate-400">{s.label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Features */}
      <Section eyebrow="Why Velos" title="Quantitative, label-free, and fast" intro="Living cells are nearly transparent. Velos QPI measures the optical phase delay they introduce and turns it into numbers biologists can use.">
        <div className="grid gap-6 md:grid-cols-2">
          {h.features.map((f) => (
            <article key={f.title} className="card">
              <h3 className="text-lg font-semibold text-white">{f.title}</h3>
              <p className="mt-2 text-slate-400">{f.body}</p>
            </article>
          ))}
        </div>
      </Section>

      {/* How it works */}
      <Section eyebrow="Workflow" title="From dish to data in four steps" className="bg-white/[0.02]">
        <ol className="grid gap-6 md:grid-cols-4">
          {h.how_it_works.map((s) => (
            <li key={s.step} className="card">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-r from-cyan-500 to-violet-600 text-sm font-bold text-white">{s.step}</div>
              <h3 className="mt-4 font-semibold text-white">{s.title}</h3>
              <p className="mt-2 text-sm text-slate-400">{s.body}</p>
            </li>
          ))}
        </ol>
        <div className="card mt-10 !p-2">
          <Image src="/images/architecture.svg" alt="Velos QPI acquisition and reconstruction architecture diagram" width={1600} height={760} className="rounded-xl" />
        </div>
      </Section>

      {/* Applications */}
      <Section eyebrow="Applications" title="Built for the questions biology labs ask">
        <div className="grid gap-6 md:grid-cols-3">
          {h.applications.map((a) => (
            <article key={a.title} className="card">
              <h3 className="text-lg font-semibold text-white">{a.title}</h3>
              <p className="mt-2 text-slate-400">{a.body}</p>
            </article>
          ))}
        </div>
        <div className="mt-12 flex flex-col items-start justify-between gap-6 rounded-2xl border border-white/10 bg-gradient-to-r from-cyan-500/10 to-violet-600/10 p-8 md:flex-row md:items-center">
          <div>
            <h3 className="text-2xl font-semibold text-white">Bring your samples. We bring the phase.</h3>
            <p className="mt-2 text-slate-400">Tell us about your assay and we will set up a demo or pilot study.</p>
          </div>
          <Link href="/contact" className="btn-primary">Request a demo</Link>
        </div>
      </Section>
    </>
  );
}
