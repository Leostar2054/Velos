"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { SITE } from "@/lib/site";
import Logo from "./Logo";

export default function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));
  return (
    <header className="sticky top-0 z-40 border-b border-white/10 bg-ink-950/80 backdrop-blur">
      <div className="container-x flex h-16 items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5" aria-label={`${SITE.company.name} home`}>
          <Logo />
          <span className="text-lg font-semibold tracking-tight text-white">
            Velos <span className="gradient-text">BioImaging</span>
          </span>
        </Link>
        <nav className="hidden items-center gap-8 md:flex" aria-label="Main">
          {SITE.nav.map((item) => (
            <Link key={item.href} href={item.href}
              className={`text-sm font-medium transition hover:text-white ${isActive(item.href) ? "text-white" : "text-slate-400"}`}
              aria-current={isActive(item.href) ? "page" : undefined}>
              {item.label}
            </Link>
          ))}
          <Link href="/contact" className="btn-primary !py-2 !px-4 text-sm">Request a demo</Link>
        </nav>
        <button className="rounded-lg p-2 text-slate-300 hover:bg-white/5 md:hidden" aria-expanded={open}
          aria-controls="mobile-menu" aria-label="Toggle menu" onClick={() => setOpen(!open)}>
          <svg className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            {open ? <path strokeLinecap="round" d="M6 6l12 12M6 18L18 6" /> : <path strokeLinecap="round" d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </div>
      {open && (
        <div id="mobile-menu" className="border-t border-white/10 md:hidden">
          <nav className="container-x flex flex-col gap-1 py-3" aria-label="Mobile">
            {SITE.nav.map((item) => (
              <Link key={item.href} href={item.href} onClick={() => setOpen(false)}
                className={`rounded-lg px-3 py-2 text-base ${isActive(item.href) ? "bg-white/5 text-white" : "text-slate-300"}`}>
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
