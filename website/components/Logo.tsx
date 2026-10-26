export default function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="velos-g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#22d3ee" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <path d="M8 12h12l12 26 12-26h12L36 56h-8L8 12z" fill="url(#velos-g)" />
      <circle cx="32" cy="20" r="5" fill="#0b1020" stroke="url(#velos-g)" strokeWidth="2.5" />
    </svg>
  );
}
