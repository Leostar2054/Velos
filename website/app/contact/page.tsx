import type { Metadata } from "next";
import { PageHeader, Section } from "@/components/Section";
import ContactForm from "@/components/ContactForm";
import { CONTACT_EMAIL, SITE } from "@/lib/site";

export const metadata: Metadata = {
  title: "Contact",
  description: "Request a Velos QPI demo, pilot study or collaboration.",
  alternates: { canonical: "/contact/" },
};

export default function ContactPage() {
  const c = SITE.contact;
  return (
    <>
      <PageHeader eyebrow="Contact" title={c.headline} intro={c.intro} />
      <Section>
        <div className="grid gap-10 lg:grid-cols-5">
          <div className="lg:col-span-3"><ContactForm /></div>
          <aside className="space-y-6 lg:col-span-2">
            {CONTACT_EMAIL && (
              <div className="card">
                <h2 className="font-semibold text-white">Email</h2>
                <a className="mt-2 block text-cyan-300 hover:underline" href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>
              </div>
            )}
            <div className="card">
              <h2 className="font-semibold text-white">Address</h2>
              <p className="mt-2 text-slate-400">{SITE.company.address}</p>
            </div>
            <div className="card">
              <h2 className="font-semibold text-white">What to include</h2>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-400">
                <li>Sample type (cell line, fixed/live, 2D/3D)</li>
                <li>Assay and readouts of interest</li>
                <li>Current imaging workflow and pain points</li>
                <li>Timeline and location</li>
              </ul>
            </div>
          </aside>
        </div>
      </Section>
    </>
  );
}
