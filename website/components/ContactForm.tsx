"use client";

import { useState } from "react";
import { API_URL, SITE } from "@/lib/site";

type State = { status: "idle" | "sending" | "sent" | "error"; message?: string };

export default function ContactForm() {
  const [state, setState] = useState<State>({ status: "idle" });

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = e.currentTarget;
    const data = new FormData(form);
    const payload = {
      name: String(data.get("name") ?? ""),
      email: String(data.get("email") ?? ""),
      organisation: String(data.get("organisation") ?? ""),
      message: `[${data.get("topic")}] ${data.get("message") ?? ""}`,
      website: String(data.get("website") ?? ""),   // honeypot
    };
    setState({ status: "sending" });
    try {
      const res = await fetch(`${API_URL}/api/v1/contact`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setState({ status: "sent" });
      form.reset();
    } catch (err) {
      setState({ status: "error", message: err instanceof Error ? err.message : "unknown error" });
    }
  }

  if (state.status === "sent") {
    return (
      <div className="card" role="status">
        <h3 className="text-lg font-semibold text-white">Thank you — message received.</h3>
        <p className="mt-2 text-slate-400">We will reply to the email address you provided, usually within two business days.</p>
      </div>
    );
  }

  return (
    <form onSubmit={onSubmit} className="card space-y-5" noValidate={false}>
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <label htmlFor="name" className="label">Name</label>
          <input id="name" name="name" required maxLength={120} className="input" placeholder="Dr. Jane Kim" />
        </div>
        <div>
          <label htmlFor="email" className="label">Email</label>
          <input id="email" name="email" type="email" required maxLength={200} className="input" placeholder="jane@university.ac.kr" />
        </div>
      </div>
      <div className="grid gap-5 sm:grid-cols-2">
        <div>
          <label htmlFor="organisation" className="label">Organisation</label>
          <input id="organisation" name="organisation" maxLength={200} className="input" placeholder="Lab / company" />
        </div>
        <div>
          <label htmlFor="topic" className="label">Topic</label>
          <select id="topic" name="topic" className="input">
            {SITE.contact.topics.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
      </div>
      <div>
        <label htmlFor="message" className="label">Message</label>
        <textarea id="message" name="message" required maxLength={4000} rows={5} className="input" placeholder="Tell us about your samples, assays and timeline." />
      </div>
      {/* Honeypot: hidden from humans, filled by bots */}
      <div className="hidden" aria-hidden="true">
        <label htmlFor="website">Website</label>
        <input id="website" name="website" tabIndex={-1} autoComplete="off" />
      </div>
      <div className="flex flex-wrap items-center gap-4">
        <button type="submit" className="btn-primary" disabled={state.status === "sending"}>
          {state.status === "sending" ? "Sending…" : "Send message"}
        </button>
        {state.status === "error" && (
          <p className="text-sm text-rose-300" role="alert">
            Could not send ({state.message}). Please email us directly at{" "}
            <a className="underline" href={`mailto:${SITE.company.email}`}>{SITE.company.email}</a>.
          </p>
        )}
      </div>
      <p className="text-xs text-slate-500">We use your details only to respond to your enquiry.</p>
    </form>
  );
}
