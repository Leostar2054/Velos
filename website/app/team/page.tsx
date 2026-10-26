import type { Metadata } from "next";
import { PageHeader, Section } from "@/components/Section";
import { SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Team",
  description: "The Velos BioImaging team: optics, computational imaging, systems and biomedical researchers from GIST.",
  alternates: { canonical: "/team/" },
};

export default function TeamPage() {
  const t = SITE.team;
  return (
    <>
      <PageHeader eyebrow="Team" title={t.headline} intro={t.intro} />
      <Section>
        <ul className="grid gap-6 md:grid-cols-2">
          {t.members.map((m) => (
            <li key={m.name} className="card flex gap-5">
              <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500 to-violet-600 text-lg font-bold text-white" aria-hidden="true">
                {m.initials}
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">{m.name}</h2>
                <p className="text-sm text-cyan-300">{m.role}</p>
                <p className="mt-3 text-sm text-slate-400">{m.bio}</p>
              </div>
            </li>
          ))}
        </ul>
        <p className="mt-10 text-sm text-slate-500">{SITE.company.affiliation}</p>
      </Section>
    </>
  );
}
