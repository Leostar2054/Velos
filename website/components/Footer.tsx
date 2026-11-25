import Link from "next/link";
import { CONTACT_EMAIL, SITE } from "@/lib/site";
import Logo from "./Logo";

export default function Footer() {
  return (
    <footer className="mt-24 border-t border-white/10">
      <div className="container-x grid gap-10 py-12 md:grid-cols-3">
        <div>
          <div className="flex items-center gap-2.5">
            <Logo className="h-7 w-7" />
            <span className="font-semibold text-white">{SITE.company.name}</span>
          </div>
          <p className="mt-3 max-w-sm text-sm text-slate-400">{SITE.company.tagline}</p>
          <p className="mt-3 text-xs text-slate-400">{SITE.company.affiliation}</p>
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white">Navigate</h3>
          <ul className="mt-3 space-y-2 text-sm text-slate-400">
            {SITE.nav.map((i) => (
              <li key={i.href}><Link className="hover:text-white" href={i.href}>{i.label}</Link></li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="text-sm font-semibold text-white">Contact</h3>
          <address className="mt-3 space-y-2 text-sm not-italic text-slate-400">
            {CONTACT_EMAIL && <p><a className="hover:text-white" href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a></p>}
            <p>{SITE.company.address}</p>
          </address>
        </div>
      </div>
      <div className="border-t border-white/10">
        <div className="container-x flex flex-col items-start justify-between gap-2 py-5 text-xs text-slate-400 sm:flex-row">
          <span>{SITE.footer.note}</span>
          <span>{SITE.company.status_note}</span>
        </div>
      </div>
    </footer>
  );
}
