import { Link } from "wouter";
import { STRINGS, type Lang } from "@/i18n/strings";
export default function Placeholder({ title, lang, milestone }: { title: string; lang: Lang; milestone: string }) {
  const t = STRINGS[lang];
  return (
    <div className="h-full w-full flex items-center justify-center bg-bg">
      <div className="panel p-6 w-[520px]">
        <div className="label">{milestone}</div>
        <h1 className="text-[18px] font-semibold mt-1">{title}</h1>
        <p className="text-[13px] text-muted mt-2">{t.nextMilestone}</p>
        <Link href="/" className="btn btn-primary mt-4 inline-flex">← {t.title}</Link>
      </div>
    </div>
  );
}
