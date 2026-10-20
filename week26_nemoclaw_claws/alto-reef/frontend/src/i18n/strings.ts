export type Lang = "en" | "th";
const en = {
  title: "Alto Reef", subtitle: "NemoClaw profiles in shared sandboxes",
  refresh: "Refresh", resetBooth: "Reset booth", kiosk: "Kiosk", buildClaw: "Build a Claw", labMode: "Lab mode", manager: "Manager",
  gateway: "Gateway", inference: "Inference", healthy: "Healthy", unreachable: "Unreachable", degraded: "Degraded",
  comms: "Comms stream", chat: "Chat", activity: "Activity", history: "History", entries: "entries",
  askPlaceholder: "Ask the team…", ask: "Ask", selected: "selected", none: "none",
  bannerMock: "MOCK DATA — no OpenShell gateway detected on this host. Values are synthetic and watermarked.",
  bannerGatewayDown: "Reef is partially active — OpenShell gateway unreachable. Sandbox state shown is last-seen, not live.",
  bannerInferenceDown: "LLM unreachable — inference provider is not answering. Runs will fail until it is back.",
  openWorkbench: "Open Workbench", unassigned: "Unassigned claws", dropHint: "Click a claw, then click a sandbox. Drag also works.",
  lastSeen: "last seen", claws: "claws", policies: "policies", noSandbox: "not in a sandbox", remove: "Remove",
  nextMilestone: "This screen lands in the next milestone. The Reef map is live.", sseLive: "live", ssePoll: "polling", sseDown: "offline",
};
const th: typeof en = {
  ...en,
  subtitle: "โปรไฟล์ NemoClaw ในแซนด์บ็อกซ์ที่ใช้ร่วมกัน", refresh: "รีเฟรช", resetBooth: "รีเซ็ตบูธ", kiosk: "คีออสก์", buildClaw: "สร้าง Claw",
  labMode: "โหมดแล็บ", manager: "ตัวจัดการ", gateway: "เกตเวย์", inference: "การอนุมาน", healthy: "ปกติ", unreachable: "ติดต่อไม่ได้", degraded: "เสื่อมสภาพ",
  comms: "สายสื่อสาร", chat: "แชท", activity: "กิจกรรม", history: "ประวัติ", entries: "รายการ", askPlaceholder: "ถามทีม…", ask: "ถาม", selected: "เลือกแล้ว", none: "ไม่มี",
  bannerMock: "ข้อมูลจำลอง — ไม่พบ OpenShell gateway บนเครื่องนี้ ค่าที่แสดงเป็นค่าสังเคราะห์",
  bannerGatewayDown: "รีฟทำงานบางส่วน — ติดต่อ OpenShell gateway ไม่ได้ สถานะที่แสดงเป็นค่าล่าสุดที่เห็น ไม่ใช่ค่าสด",
  bannerInferenceDown: "LLM ติดต่อไม่ได้ — ผู้ให้บริการอนุมานไม่ตอบ การรันจะล้มเหลวจนกว่าจะกลับมา",
  openWorkbench: "เปิด Workbench", unassigned: "Claw ที่ยังไม่ได้มอบหมาย", dropHint: "คลิก claw แล้วคลิกแซนด์บ็อกซ์ หรือลากวางก็ได้",
  lastSeen: "เห็นล่าสุด", claws: "claws", policies: "นโยบาย", noSandbox: "ไม่อยู่ในแซนด์บ็อกซ์", remove: "ลบออก",
  nextMilestone: "หน้านี้จะมาในไมล์สโตนถัดไป แผนที่รีฟใช้งานได้แล้ว", sseLive: "สด", ssePoll: "โพล", sseDown: "ออฟไลน์",
};
export const STRINGS: Record<Lang, typeof en> = { en, th };
