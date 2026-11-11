import { useId } from "react";

/** Velos mark. Each instance gets its own gradient id (the nav and the footer both render a
 *  logo; a shared id is invalid HTML and breaks the gradient when one instance is hidden). */
export default function Logo({ className = "h-8 w-8" }: { className?: string }) {
  const gid = `velos-g-${useId().replace(/:/g, "")}`;
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <path d="M8 12h12l12 26 12-26h12L36 56h-8L8 12z" fill={`url(#${gid})`} />
      <circle cx="32" cy="20" r="5" fill="#0b1020" stroke={`url(#${gid})`} strokeWidth="2.5" />
    </svg>
  );
}
