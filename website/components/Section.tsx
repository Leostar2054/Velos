import type { ReactNode } from "react";

export function Section({ id, eyebrow, title, intro, children, className = "" }:
  { id?: string; eyebrow?: string; title?: string; intro?: string; children: ReactNode; className?: string }) {
  return (
    <section id={id} className={`py-16 sm:py-20 ${className}`}>
      <div className="container-x">
        {(eyebrow || title || intro) && (
          <div className="mb-10 max-w-3xl">
            {eyebrow && <p className="eyebrow">{eyebrow}</p>}
            {title && <h2 className="mt-2 text-3xl font-semibold tracking-tight text-white sm:text-4xl">{title}</h2>}
            {intro && <p className="mt-4 text-lg text-slate-400">{intro}</p>}
          </div>
        )}
        {children}
      </div>
    </section>
  );
}

export function PageHeader({ eyebrow, title, intro }: { eyebrow: string; title: string; intro: string }) {
  return (
    <div className="relative overflow-hidden border-b border-white/10">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_60%_at_50%_0%,rgba(34,211,238,0.12),transparent_70%)]" />
      <div className="container-x relative py-16 sm:py-20">
        <p className="eyebrow">{eyebrow}</p>
        <h1 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight text-white sm:text-5xl">{title}</h1>
        <p className="mt-5 max-w-3xl text-lg text-slate-400">{intro}</p>
      </div>
    </div>
  );
}
