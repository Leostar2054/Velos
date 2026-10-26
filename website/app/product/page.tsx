import type { Metadata } from "next";
import Image from "next/image";
import { PageHeader, Section } from "@/components/Section";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Product — Velos QPI and SPARK",
  description: "Velos QPI compact 630 nm quantitative phase imaging hardware and the SPARK reconstruction platform: physics-based, sparse-scanning, AI-assisted and hybrid modes.",
  alternates: { canonical: "/product/" },
};

export default function ProductPage() {
  const p = SITE.product;
  return (
    <>
      <PageHeader eyebrow="Product" title={p.headline} intro={p.intro} />

      <Section eyebrow="Hardware" title="Velos QPI imaging module" intro="A compact optical acquisition module built around a 630 nm visible-light source, beam-shaping optics, a controlled XY sample stage and a CMOS camera.">
        <div className="overflow-hidden rounded-2xl border border-white/10">
          <table className="w-full text-left text-sm">
            <thead className="bg-white/[0.04] text-slate-300">
              <tr><th className="px-5 py-3 font-semibold">Module</th><th className="px-5 py-3 font-semibold">Design</th></tr>
            </thead>
            <tbody className="divide-y divide-white/10">
              {p.hardware.map((r) => (
                <tr key={r.module}><td className="px-5 py-3 font-medium text-white">{r.module}</td><td className="px-5 py-3 text-slate-400">{r.detail}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section eyebrow="Software" title="SPARK reconstruction platform" className="bg-white/[0.02]"
        intro="One data model, one calibration, one quality-control workflow — and the right reconstruction engine for each sample, speed requirement and level of quantitative reliability.">
        <div className="grid gap-6 md:grid-cols-2">
          {p.modes.map((m) => (
            <article key={m.name} className="card">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-lg font-semibold text-white">{m.name}</h3>
                <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-xs font-medium text-cyan-200">{m.status}</span>
              </div>
              <p className="mt-3 text-slate-400">{m.body}</p>
            </article>
          ))}
        </div>
        <div className="card mt-10 !p-2">
          <Image src="/images/spark-workflow.jpg" alt="Recommended SPARK software workflow diagram" width={1600} height={855} className="rounded-xl" />
        </div>
        <div className="mt-10 grid gap-6 md:grid-cols-4">
          {p.software_platform.map((f) => (
            <div key={f.title} className="card">
              <h3 className="font-semibold text-white">{f.title}</h3>
              <p className="mt-2 text-sm text-slate-400">{f.body}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section eyebrow="Performance" title="Development targets" intro="All targets require experimental confirmation and are reported with their validation status.">
        <dl className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {p.targets.map((t) => (
            <div key={t.metric} className="card">
              <dt className="text-sm text-slate-400">{t.metric}</dt>
              <dd className="mt-1 text-xl font-semibold text-white">{t.value}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-8 max-w-3xl text-slate-400">{p.validation_note}</p>
        <div className="card mt-8 !p-2">
          <Image src="/images/validation-roadmap.jpg" alt="Progressive sample and validation roadmap" width={1600} height={674} className="rounded-xl" />
        </div>
      </Section>
    </>
  );
}
