import Script from "next/script";
import { GA_ID, PLAUSIBLE_DOMAIN } from "@/lib/site";

/**
 * Analytics hook. Nothing is loaded unless an environment variable is set at build time:
 *   NEXT_PUBLIC_PLAUSIBLE_DOMAIN=velosbioimaging.com   → Plausible (cookie-less, GDPR friendly)
 *   NEXT_PUBLIC_GA_ID=G-XXXXXXX                        → Google Analytics 4
 */
export default function Analytics() {
  return (
    <>
      {PLAUSIBLE_DOMAIN && (
        <Script defer data-domain={PLAUSIBLE_DOMAIN} src="https://plausible.io/js/script.js" strategy="afterInteractive" />
      )}
      {GA_ID && (
        <>
          <Script src={`https://www.googletagmanager.com/gtag/js?id=${GA_ID}`} strategy="afterInteractive" />
          <Script id="ga4" strategy="afterInteractive">
            {`window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments);}gtag('js',new Date());gtag('config','${GA_ID}',{anonymize_ip:true});`}
          </Script>
        </>
      )}
    </>
  );
}
