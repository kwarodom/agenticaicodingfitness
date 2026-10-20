const joints = [
  { id: "shoulder_pan", label: "Shoulder pan", positive: "Q", negative: "A" },
  { id: "shoulder_lift", label: "Shoulder lift", positive: "W", negative: "S" },
  { id: "elbow_flex", label: "Elbow flex", positive: "E", negative: "D" },
  { id: "wrist_flex", label: "Wrist flex", positive: "R", negative: "F" },
  { id: "wrist_roll", label: "Wrist roll", positive: "T", negative: "G" },
  { id: "gripper", label: "Gripper", positive: "Y", negative: "H" },
];

const llmProfiles = {
  openrouter: {
    label: "OpenRouter",
    model: "openrouter/free",
    models: ["openrouter/free"],
    help: "One key gives access to many hosted models, including a capability-aware free route.",
  },
  openai: {
    label: "OpenAI",
    model: "",
    models: [],
    help: "Connect directly to OpenAI with your OpenAI API key.",
  },
  anthropic: {
    label: "Anthropic",
    model: "claude-sonnet-5-5",
    models: ["claude-sonnet-5-5", "claude-opus-4-8", "claude-haiku-4-5-20251001"],
    help: "Connect directly to Claude through the Anthropic Messages API.",
  },
  glm: {
    label: "GLM",
    model: "glm-5.2",
    models: ["glm-5.2", "glm-4.6", "glm-4.5", "glm-4.5-air"],
    help: "Connect to Z.ai's GLM coding endpoint with your Z.ai API key.",
  },
  custom: {
    label: "Custom",
    model: "openthaigpt-thaillm-8b-instruct-v7.2",
    models: ["openthaigpt-thaillm-8b-instruct-v7.2"],
    help: "Use an OpenAI-compatible endpoint such as Ollama, LM Studio, or another gateway.",
  },
};

const llmPreferencesKey = "agentic-robotics.llm-preferences";
const llmSecretsKey = "agentic-robotics.llm-keys";
const hardwarePreferencesKey = "agentic-robotics.hardware-preferences";

function storedJson(key) {
  try { return JSON.parse(localStorage.getItem(key) || "{}"); } catch (_) { return {}; }
}

const savedLlmPreferences = storedJson(llmPreferencesKey);
const llmModels = Object.fromEntries(
  Object.entries(llmProfiles).map(([provider, profile]) => [
    provider,
    savedLlmPreferences.models?.[provider] ?? profile.model,
  ]),
);
const llmBaseUrls = { custom: savedLlmPreferences.baseUrls?.custom || "http://thaillm.or.th/api/v1" };
const llmKeys = { openrouter: "", openai: "", anthropic: "", glm: "", custom: "", ...storedJson(llmSecretsKey) };
let llmProvider = llmProfiles[savedLlmPreferences.provider]
  ? savedLlmPreferences.provider
  : "custom";
let dialogLlmProvider = llmProvider;

const keyMap = new Map();
for (const joint of joints) {
  keyMap.set(joint.positive.toLowerCase(), { joint: joint.id, direction: 1 });
  keyMap.set(joint.negative.toLowerCase(), { joint: joint.id, direction: -1 });
}

const heldKeys = new Set();
let latestState = null;
let toastTimer = null;
let lastError = null;
let artifactSignature = null;
let memorySignature = null;
let reconstructionSignature = null;
let chatSignature = null;
let hardwareSignature = null;
let discoveredHardware = null;
let robotSetupStep = 1;
let robotSetupScanComplete = false;
let physicalCalibrationStep = 1;
let lastPhysicalPlanId = null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value).replace(
    /[&<>"]/g,
    (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character],
  );
}

function renderInlineMarkdown(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*\n]+)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

// Escape first, then apply a small markdown subset, so model output can format
// but can never inject HTML.
function renderMarkdown(content) {
  const blocks = escapeHtml(String(content).trim()).split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n").filter((line) => line.trim() !== "");
    const bulleted = lines.length && lines.every((line) => /^\s*[-*]\s+/.test(line));
    const numbered = lines.length && lines.every((line) => /^\s*\d+\.\s+/.test(line));
    if (bulleted || numbered) {
      const items = lines
        .map((line) => `<li>${renderInlineMarkdown(line.replace(/^\s*(?:[-*]|\d+\.)\s+/, ""))}</li>`)
        .join("");
      return numbered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
    }
    return `<p>${renderInlineMarkdown(block).replace(/\n/g, "<br>")}</p>`;
  }).join("");
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.className = "toast"; }, 2800);
}

function currentLlmSettings() {
  return {
    provider: llmProvider,
    model: llmModels[llmProvider].trim(),
    api_key: llmKeys[llmProvider].trim(),
    base_url: llmProvider === "custom" ? llmBaseUrls.custom.trim() : null,
  };
}

function llmConfigurationError(settings = currentLlmSettings()) {
  if (!settings.model) return "Enter a model ID.";
  if (settings.provider !== "custom" && !settings.api_key) {
    return `Enter your ${llmProfiles[settings.provider].label} API key.`;
  }
  if (settings.provider === "custom" && !settings.base_url) {
    return "Enter the OpenAI-compatible base URL.";
  }
  return null;
}

function renderLlmBadge() {
  const settings = currentLlmSettings();
  const button = $("#chat-model");
  const model = settings.model || "Model settings";
  button.textContent = settings.model
    ? `${llmProfiles[settings.provider].label} / ${model.split("/").pop()}`
    : "Model settings";
  button.title = settings.model
    ? `${llmProfiles[settings.provider].label} · ${settings.model} · click to change`
    : "Configure the robot copilot model";
}

function populateLlmDialog() {
  dialogLlmProvider = llmProvider;
  $("#llm-provider").value = llmProvider;
  $("#llm-model").value = llmModels[llmProvider];
  $("#llm-api-key").value = llmKeys[llmProvider];
  $("#llm-base-url").value = llmBaseUrls.custom;
  $("#remember-llm-key").checked = Boolean(storedJson(llmSecretsKey)[llmProvider]);
  $("#llm-api-key").type = "password";
  $("#toggle-llm-key").textContent = "Show";
  renderLlmProviderFields();
}

function renderLlmProviderFields() {
  const profile = llmProfiles[dialogLlmProvider];
  $("#llm-provider-help").textContent = profile.help;
  $("#llm-base-url-field").hidden = dialogLlmProvider !== "custom";
  $("#llm-key-optional").textContent = dialogLlmProvider === "custom" ? "· optional" : "";
  $("#llm-model-options").innerHTML = profile.models
    .map((model) => `<option value="${escapeHtml(model)}"></option>`)
    .join("");
}

function changeLlmProvider(provider) {
  llmModels[dialogLlmProvider] = $("#llm-model").value;
  llmKeys[dialogLlmProvider] = $("#llm-api-key").value;
  dialogLlmProvider = provider;
  $("#llm-model").value = llmModels[provider];
  $("#llm-api-key").value = llmKeys[provider];
  $("#remember-llm-key").checked = Boolean(storedJson(llmSecretsKey)[provider]);
  renderLlmProviderFields();
}

function saveLlmSettings() {
  llmModels[dialogLlmProvider] = $("#llm-model").value.trim();
  llmKeys[dialogLlmProvider] = $("#llm-api-key").value.trim();
  llmBaseUrls.custom = $("#llm-base-url").value.trim();
  llmProvider = dialogLlmProvider;
  const error = llmConfigurationError();
  if (error) {
    showToast(error, true);
    return;
  }

  localStorage.setItem(llmPreferencesKey, JSON.stringify({
    provider: llmProvider,
    models: llmModels,
    baseUrls: llmBaseUrls,
  }));
  const savedKeys = storedJson(llmSecretsKey);
  if ($("#remember-llm-key").checked && llmKeys[llmProvider]) {
    savedKeys[llmProvider] = llmKeys[llmProvider];
  } else {
    delete savedKeys[llmProvider];
  }
  if (Object.keys(savedKeys).length) {
    localStorage.setItem(llmSecretsKey, JSON.stringify(savedKeys));
  } else {
    localStorage.removeItem(llmSecretsKey);
  }
  $("#llm-dialog").close();
  renderLlmBadge();
  showToast(`${llmProfiles[llmProvider].label} · ${llmModels[llmProvider]} selected`);
}

function buildJointControls() {
  const markup = joints.map((joint) => `
    <div class="joint-control">
      <button class="hold-button" data-joint="${joint.id}" data-direction="-1" aria-label="Move ${joint.label} negative">${joint.negative} −</button>
      <span><b>${joint.label}</b><small>HOLD TO MOVE</small></span>
      <button class="hold-button" data-joint="${joint.id}" data-direction="1" aria-label="Move ${joint.label} positive">+ ${joint.positive}</button>
    </div>
  `).join("");
  for (const selector of ["#joint-controls", "#calibration-joint-controls"]) {
    const container = $(selector);
    if (container) container.innerHTML = markup;
  }

  for (const button of $$(".hold-button")) {
    const setActive = async (active) => {
      button.classList.toggle("active", active);
      try {
        await api("/api/manual/control", {
          method: "POST",
          body: JSON.stringify({
            joint: button.dataset.joint,
            direction: Number(button.dataset.direction),
            active,
          }),
        });
      } catch (error) {
        button.classList.remove("active");
        showToast(error.message, true);
      }
    };
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      setActive(true);
    });
    for (const eventName of ["pointerup", "pointercancel", "lostpointercapture"]) {
      button.addEventListener(eventName, () => setActive(false));
    }
  }
}

async function releaseAll() {
  heldKeys.clear();
  $$(".hold-button.active").forEach((button) => button.classList.remove("active"));
  try { await api("/api/manual/release", { method: "POST" }); } catch (_) {}
}

function bindKeyboard() {
  window.addEventListener("keydown", (event) => {
    const control = keyMap.get(event.key.toLowerCase());
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (!control || typing || heldKeys.has(event.key.toLowerCase())) return;
    event.preventDefault();
    heldKeys.add(event.key.toLowerCase());
    api("/api/manual/control", {
      method: "POST",
      body: JSON.stringify({ ...control, active: true }),
    }).catch((error) => showToast(error.message, true));
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    const control = keyMap.get(key);
    if (!control || !heldKeys.has(key)) return;
    event.preventDefault();
    heldKeys.delete(key);
    api("/api/manual/control", {
      method: "POST",
      body: JSON.stringify({ ...control, active: false }),
    }).catch((error) => showToast(error.message, true));
  });
  window.addEventListener("blur", releaseAll);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) releaseAll();
  });
}

function connectStream(element, camera) {
  const reconnect = () => {
    element.src = `/api/stream/${camera}?t=${Date.now()}`;
  };
  element.addEventListener("error", () => setTimeout(reconnect, 1000));
  reconnect();
}

function renderArtifacts(artifacts) {
  const signature = artifacts.map((artifact) => artifact.url).join("|");
  if (signature === artifactSignature) return;
  artifactSignature = signature;
  const gallery = $("#artifact-gallery");
  if (!artifacts.length) {
    gallery.innerHTML = '<div class="empty-state">No perception outputs yet.</div>';
    return;
  }
  gallery.innerHTML = artifacts.map((artifact) => `
    <a class="artifact-card ${artifact.live ? "live" : ""}" href="${artifact.url}" target="_blank" title="Open ${escapeHtml(artifact.title)}">
      <img src="${artifact.url}" alt="${escapeHtml(artifact.title)}" loading="lazy">
      <span>${escapeHtml(artifact.title)}</span>
    </a>
  `).join("");
}

function renderSemanticMemory(scene, labelsReady) {
  const objects = labelsReady
    ? scene.semantic?.objects || []
    : scene.geometry?.objects || [];
  const signature = JSON.stringify(objects.map((object) => [
    object.id,
    object.label,
    object.position,
  ]));
  if (signature === memorySignature) return;
  memorySignature = signature;
  const container = $("#semantic-memory");
  if (!objects.length) {
    container.innerHTML = '<span class="memory-placeholder">No objects in memory.</span>';
    return;
  }
  container.innerHTML = objects.map((object) => {
    const label = object.label || object.id;
    const position = object.position?.slice(0, 2).map((value) => Number(value).toFixed(3)).join(", ");
    const details = [object.label ? object.id : null, position ? `XY ${position} m` : null]
      .filter(Boolean)
      .join(" · ");
    return `
      <div class="semantic-object">
        <b>${escapeHtml(label)}</b>
        ${details ? `<small>${escapeHtml(details)}</small>` : ""}
      </div>
    `;
  }).join("");
}

function renderReconstructionStages(reconstruction) {
  const container = $("#reconstruction-stages");
  const signature = JSON.stringify(reconstruction);
  if (signature === reconstructionSignature) return;
  reconstructionSignature = signature;
  if (!reconstruction?.stages) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }

  const stages = [
    ["foreground_masks", "Foreground masks"],
    ["depth_prediction", "Learned depth"],
    ["table_alignment", "Metric alignment"],
    ["cross_view_consistency", "Cross-view agreement"],
    ["voxel_fusion", "Point-cloud fusion"],
    ["object_fitting", "Planner objects"],
  ];
  const detail = (name, stage = {}) => {
    if (name === "foreground_masks" && stage.views) return `${stage.views} calibrated views`;
    if (name === "depth_prediction" && stage.views) return `${stage.views} RGB images`;
    if (name === "table_alignment" && stage.plane) return "fitted to table z = 0";
    if (name === "cross_view_consistency" && stage.candidate_points !== undefined) {
      return `${stage.accepted_points.toLocaleString()} / ${stage.candidate_points.toLocaleString()} points kept`;
    }
    if (name === "voxel_fusion" && stage.fused_voxels !== undefined) {
      return `${stage.fused_voxels.toLocaleString()} metric voxels`;
    }
    if (name === "object_fitting" && stage.objects !== undefined) return `${stage.objects} objects`;
    return stage.reason || "";
  };
  const rows = stages.map(([name, label]) => {
    const stage = reconstruction.stages[name] || { status: "waiting" };
    return `
      <li class="reconstruction-stage ${escapeHtml(stage.status || "waiting")}">
        <i aria-hidden="true"></i>
        <b>${escapeHtml(label)}</b>
        <span>${escapeHtml(detail(name, stage) || stage.status || "waiting")}</span>
      </li>
    `;
  }).join("");

  const alignments = Object.entries(reconstruction.views || {});
  const alignmentDetails = alignments.length ? `
    <details class="alignment-details">
      <summary>Table alignment by camera view</summary>
      <div>
        ${alignments.map(([name, record]) => `
          <p>
            <b>${escapeHtml(name.replace(/^survey_/, "").replaceAll("_", " "))}</b>
            <span>scale ${Number(record.scale).toFixed(3)} · offset ${Number(record.offset_meters * 1000).toFixed(1)} mm · RMSE ${Number(record.table_rmse_meters * 1000).toFixed(1)} mm</span>
          </p>
        `).join("")}
      </div>
    </details>
  ` : "";
  const registrations = Object.entries(
    reconstruction.stages?.foreground_masks?.table_plane_registration || {},
  );
  const registrationDetails = registrations.length ? `
    <details class="alignment-details">
      <summary>Empty/occupied camera-pose registration</summary>
      <div>
        ${registrations.map(([name, record]) => `
          <p>
            <b>${escapeHtml(name.replace(/^survey_/, "").replaceAll("_", " "))}</b>
            <span>${Number(record.translation_delta_meters * 1000).toFixed(1)} mm · ${Number(record.rotation_delta_degrees).toFixed(1)}° · ${Number(record.valid_pixel_fraction * 100).toFixed(0)}% overlap · ${Number(record.foreground_pixel_fraction * 100).toFixed(1)}% foreground</span>
          </p>
        `).join("")}
      </div>
    </details>
  ` : "";

  container.innerHTML = `
    <div class="reconstruction-heading">
      <b>Reconstruction</b>
      <span>${escapeHtml(reconstruction.depth_model || "metric depth")} · ${escapeHtml(reconstruction.device || "auto")}</span>
    </div>
    <ol>${rows}</ol>
    ${registrationDetails}
    ${alignmentDetails}
  `;
  container.hidden = false;
}

function renderHardwareSetup(hardware) {
  if (!hardware) return;
  discoveredHardware = hardware;
  const signature = JSON.stringify(hardware);
  if (signature === hardwareSignature) return;
  hardwareSignature = signature;

  const ports = hardware.ports || [];
  const cameras = hardware.cameras || [];
  const calibrations = hardware.calibrations || [];
  $("#serial-port-options").innerHTML = ports.map((port) =>
    `<option value="${escapeHtml(port.device)}">${escapeHtml(port.description)}</option>`,
  ).join("");
  $("#camera-index-options").innerHTML = cameras.map((camera, index) => {
    const cameraId = camera.id ?? camera.index_or_path ?? index;
    return `<option value="${escapeHtml(cameraId)}"></option>`;
  }).join("");

  const followers = calibrations.filter((record) => record.role === "follower");
  const leaders = calibrations.filter((record) => record.role === "leader");
  $("#follower-id-options").innerHTML = followers
    .map((record) => `<option value="${escapeHtml(record.id)}"></option>`)
    .join("");
  $("#leader-id-options").innerHTML = leaders
    .map((record) => `<option value="${escapeHtml(record.id)}"></option>`)
    .join("");

  if (ports.length === 1 && !ports.some((port) => port.device === $("#real-port").value.trim())) {
    $("#real-port").value = ports[0].device;
  }
  const cameraIds = cameras.map((camera, index) => String(camera.id ?? camera.index_or_path ?? index));
  if (cameraIds.length === 1 && !cameraIds.includes($("#real-camera-index").value)) {
    $("#real-camera-index").value = cameraIds[0];
  }
  if (followers.length === 1 && !calibrationRecord("follower", $("#real-robot-id").value.trim())) {
    $("#real-robot-id").value = followers[0].id;
  }
  if (leaders.length === 1 && !$("#real-leader-id").value.trim()) {
    $("#real-leader-id").value = leaders[0].id;
  }

  const installed = hardware.lerobot?.installed;
  const version = hardware.lerobot?.version;
  if (!hardware.scanned) {
    $("#hardware-discovery-status").textContent = installed
      ? `LeRobot ${version} ready · scan connected hardware`
      : "LeRobot is not installed in this environment";
  } else {
    $("#hardware-discovery-status").textContent = installed
      ? `LeRobot ${version} · ${ports.length} serial devices · ${cameras.length} cameras`
      : "LeRobot is not installed in this environment";
  }

  const discoveryError = $("#hardware-discovery-error");
  if (!installed) {
    discoveryError.textContent = "Install the root requirements in this project’s active .venv, then restart the portal.";
    discoveryError.hidden = false;
  } else if (hardware.camera_error) {
    discoveryError.textContent = `Camera scan failed: ${hardware.camera_error}`;
    discoveryError.hidden = false;
  } else {
    discoveryError.hidden = true;
    discoveryError.textContent = "";
  }
  renderSelectedCalibration();
  renderRobotSetupReview();
}

function calibrationRecord(role, id) {
  return (discoveredHardware?.calibrations || []).find(
    (item) => item.role === role && item.id === id,
  );
}

function renderSelectedCalibration() {
  const followerId = $("#real-robot-id").value.trim();
  const follower = calibrationRecord("follower", followerId);
  const followerSummary = $("#calibration-summary");
  if (!follower) {
    followerSummary.textContent = followerId
      ? `No saved follower calibration found for “${followerId}”.`
      : "Choose a follower calibration ID.";
    followerSummary.className = "calibration-summary error";
    $("#joint-limit-list").innerHTML = "";
  } else {
    followerSummary.textContent = `Follower calibration found · ${follower.id}`;
    followerSummary.className = "calibration-summary success";
    $("#joint-limit-list").innerHTML = Object.entries(follower.limits).map(([joint, limit]) => `
      <div>
        <span>${escapeHtml(joint.replaceAll("_", " "))}</span>
        <b>${escapeHtml(limit.min)} to ${escapeHtml(limit.max)} ${escapeHtml(limit.unit)}</b>
      </div>
    `).join("");
  }

  const leaderId = $("#real-leader-id").value.trim();
  const leader = calibrationRecord("leader", leaderId);
  const leaderSummary = $("#leader-calibration-summary");
  if (!leaderId) {
    leaderSummary.textContent = "Leader calibration is optional.";
    leaderSummary.className = "calibration-summary secondary";
  } else if (leader) {
    leaderSummary.textContent = `Leader calibration found · ${leader.id}`;
    leaderSummary.className = "calibration-summary secondary success";
  } else {
    leaderSummary.textContent = `No saved leader calibration found for “${leaderId}” · optional for this connection.`;
    leaderSummary.className = "calibration-summary secondary";
  }
  renderRobotSetupReview();
}

function setRobotSetupMessage(message = "", isError = true) {
  const element = $("#robot-setup-message");
  element.textContent = message;
  element.hidden = !message;
  element.className = `robot-setup-message${isError ? "" : " info"}`;
}

function cameraRotationLabel() {
  return $("#real-camera-rotation").selectedOptions[0]?.textContent || "Unknown orientation";
}

function renderRobotSetupReview() {
  const followerPort = $("#real-port").value.trim();
  const followerId = $("#real-robot-id").value.trim();
  const cameraIndex = $("#real-camera-index").value;
  const width = $("#real-camera-width").value;
  const height = $("#real-camera-height").value;
  const fps = $("#real-camera-fps").value;
  const leaderPort = $("#real-leader-port").value.trim();
  const leaderId = $("#real-leader-id").value.trim();

  $("#setup-review-follower").textContent = followerPort || "Not selected";
  $("#setup-review-calibration").textContent = followerId
    ? `${followerId}${calibrationRecord("follower", followerId) ? " · found" : " · missing"}`
    : "Not selected";
  $("#setup-review-camera").textContent = cameraIndex === ""
    ? "Not selected"
    : `Camera ${cameraIndex} · ${width} × ${height} at ${fps} FPS · ${cameraRotationLabel()}`;
  $("#setup-review-leader").textContent = leaderPort
    ? `${leaderPort}${leaderId ? ` · ${leaderId}` : ""} · saved for later teleoperation`
    : "Optional · not configured";
}

function robotSetupValidation(step) {
  if (step === 1) {
    const confirmed = [
      "#setup-confirm-built",
      "#setup-confirm-secure",
      "#setup-confirm-stop",
    ].every((selector) => $(selector).checked);
    return confirmed ? null : "Confirm all three physical safety checks before continuing.";
  }

  if (step === 2) {
    if (!robotSetupScanComplete || !discoveredHardware?.scanned) {
      return "Scan the connected hardware before assigning devices.";
    }
    if (!discoveredHardware.lerobot?.installed) {
      return "LeRobot is missing from the Python environment running this portal.";
    }
    const followerPort = $("#real-port").value.trim();
    if (!followerPort) return "Choose the follower USB port.";
    const ports = discoveredHardware.ports || [];
    if (ports.length && !ports.some((port) => port.device === followerPort)) {
      return "The selected follower port is not present in the latest hardware scan.";
    }
    const cameraIndex = Number($("#real-camera-index").value);
    if (!Number.isInteger(cameraIndex) || cameraIndex < 0) return "Choose a valid wrist-camera index.";
    const cameras = discoveredHardware.cameras || [];
    if (cameras.length) {
      const selected = String(cameraIndex);
      const exists = cameras.some((camera, index) =>
        String(camera.id ?? camera.index_or_path ?? index) === selected,
      );
      if (!exists) return "The selected wrist camera is not present in the latest hardware scan.";
    } else if (discoveredHardware.camera_error) {
      return "Resolve the camera scan error before connecting the physical robot.";
    } else {
      return "No wrist camera was found. Connect it and scan again.";
    }
    const cameraFields = [
      "#real-camera-width",
      "#real-camera-height",
      "#real-camera-fps",
    ];
    if (!cameraFields.every((selector) => $(selector).checkValidity())) {
      return "Use valid width, height, and FPS values for the wrist camera.";
    }
    return null;
  }

  if (step === 3) {
    const followerId = $("#real-robot-id").value.trim();
    if (!followerId) return "Choose the follower calibration ID.";
    if (!calibrationRecord("follower", followerId)) {
      return `No saved follower calibration matches “${followerId}”.`;
    }
    return null;
  }

  if (step === 4 && !$("#setup-confirm-connect").checked) {
    return "Confirm that the arm is supported and motion controls will remain locked.";
  }
  return null;
}

function showRobotSetupStep(step, message = "") {
  robotSetupStep = Math.max(1, Math.min(4, step));
  const labels = ["", "Prepare", "Devices", "Calibration", "Review"];
  $$("[data-setup-step]").forEach((section) => {
    section.hidden = Number(section.dataset.setupStep) !== robotSetupStep;
  });
  $$("[data-setup-progress]").forEach((item) => {
    const itemStep = Number(item.dataset.setupProgress);
    item.classList.toggle("current", itemStep === robotSetupStep);
    item.classList.toggle("complete", itemStep < robotSetupStep);
    if (itemStep === robotSetupStep) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  });
  $("#robot-setup-position").textContent = `Step ${robotSetupStep} of 4 · ${labels[robotSetupStep]}`;
  $("#robot-setup-back").hidden = robotSetupStep === 1;
  $("#robot-setup-next").hidden = robotSetupStep === 4;
  $("#connect-real-button").hidden = robotSetupStep !== 4;
  $(".robot-setup-content").scrollTop = 0;
  renderRobotSetupReview();
  setRobotSetupMessage(message);
}

function openRobotSetup() {
  robotSetupScanComplete = false;
  [
    "#setup-confirm-built",
    "#setup-confirm-secure",
    "#setup-confirm-stop",
    "#setup-confirm-connect",
  ].forEach((selector) => { $(selector).checked = false; });
  showRobotSetupStep(1);
  $("#robot-dialog").showModal();
}

function advanceRobotSetup() {
  const error = robotSetupValidation(robotSetupStep);
  if (error) {
    setRobotSetupMessage(error);
    return;
  }
  showRobotSetupStep(robotSetupStep + 1);
}

function returnToPreviousRobotSetupStep() {
  showRobotSetupStep(robotSetupStep - 1);
}

function readinessMap(readiness) {
  return new Map(readiness.map((step) => [step.number, step.ready]));
}

function renderMemoryActivity(state, readiness) {
  const runningTool = [...state.chat.messages]
    .reverse()
    .find((message) => message.role === "tool" && message.status === "running");

  let status = "No scene captured";
  const runningReconstruction = Object.entries(state.scene.reconstruction?.stages || {})
    .find(([, stage]) => stage.status === "running");
  const failedReconstruction = Object.entries(state.scene.reconstruction?.stages || {})
    .find(([, stage]) => stage.status === "error");
  const reviewReconstruction = Object.entries(state.scene.reconstruction?.stages || {})
    .find(([, stage]) => stage.status === "review");
  const fittingComplete = state.scene.reconstruction?.stages?.object_fitting?.status === "complete";
  if (runningTool) {
    status = runningTool.content;
  } else if (runningReconstruction) {
    status = `Reconstruction · ${runningReconstruction[0].replaceAll("_", " ")}`;
  } else if (failedReconstruction) {
    status = `Reconstruction stopped · ${failedReconstruction[0].replaceAll("_", " ")}`;
  } else if (readiness.get("05") && reviewReconstruction) {
    status = `Geometry built · ${reviewReconstruction[0].replaceAll("_", " ")} needs review`;
  } else if (readiness.get("05S")) {
    status = "Semantic scene ready for robot commands";
  } else if (readiness.get("05")) {
    status = "Geometry ready · semantic naming is next";
  } else if (fittingComplete) {
    status = "Geometry built · preparing object crops";
  } else if (readiness.get("04B")) {
    status = "Object scan ready · reconstruction is next";
  } else if (readiness.get("04A")) {
    status = "Calibration ready · object scan is next";
  }
  $("#memory-live-status").textContent = status;
}

function compactToolResult(result = {}) {
  const summary = {};
  for (const key of [
    "status",
    "next_required",
    "semantic_provider",
    "object_label",
    "destination",
    "safe",
    "verified",
    "plan_id",
    "robot_mode",
    "pick_xyz",
    "place_xyz",
    "reason",
  ]) {
    if (result[key] !== undefined) summary[key] = result[key];
  }
  if (result.stages) summary.stages = result.stages;
  if (result.object_ids) summary.object_ids = result.object_ids;
  if (result.objects) {
    summary.objects = result.objects.map((object) => object.label || object.id);
  }
  return summary;
}

function renderChat(state, realMode) {
  const signature = JSON.stringify([
    state.chat.messages,
    state.chat.thinking,
    realMode,
  ]);
  if (signature !== chatSignature) {
    chatSignature = signature;
    const container = $("#chat-messages");
    const messages = state.chat.messages.map((message) => {
      if (message.role === "tool") {
        const expanded = message.status !== "complete";
        const result = message.result
          ? `<div class="chat-tool-result">${escapeHtml(JSON.stringify(compactToolResult(message.result), null, 2))}</div>`
          : "";
        return `
          <details class="chat-tool" ${expanded ? "open" : ""}>
            <summary class="chat-tool-head">
              <b>${escapeHtml(message.content)}</b>
              <span class="tool-state ${escapeHtml(message.status || "running")}">${escapeHtml(message.status || "running")}</span>
            </summary>
            ${result}
          </details>
        `;
      }
      const isAssistant = message.role === "assistant";
      const body = isAssistant ? renderMarkdown(message.content) : escapeHtml(message.content);
      return `<article class="chat-message ${escapeHtml(message.role)}${isAssistant ? " markdown" : ""} ${message.status === "error" ? "error" : ""}">${body}</article>`;
    }).join("");
    const typing = state.chat.thinking
      ? '<article class="chat-message assistant chat-typing" role="status" aria-label="Copilot is typing"><i></i><i></i><i></i></article>'
      : "";
    container.innerHTML = (messages || typing)
      ? `${messages}${typing}`
      : '<div class="chat-empty">Ask for an outcome, such as “move the pen to the right.”</div>';
    container.scrollTop = container.scrollHeight;
  }

  const busy = state.workflow.busy || state.chat.thinking;
  const runningTool = [...state.chat.messages]
    .reverse()
    .find((message) => message.role === "tool" && message.status === "running");
  $("#chat-status-text").textContent = runningTool?.content
    || (busy ? "Working" : realMode ? "Physical mode · safety gated" : "Ready");
  renderLlmBadge();

  $("#chat-input").disabled = false;
  $("#chat-send").disabled = busy;
  $("#new-chat-button").disabled = busy;
  $("#chat-hint").textContent = llmConfigurationError()
      ? "Configure a model before sending."
      : realMode
        ? "Physical scans and manipulation require a separate safety-bar approval."
        : "";
}

function physicalWorkspaceSummary(workspace) {
  if (!workspace?.compatible) return "Calibrate camera and survey poses";
  if (workspace.execution_ready) return "Ready for scanning and approved manipulation";
  if (workspace.survey_ready && workspace.kinematic_validation?.status !== "pass") {
    return "Survey recorded · measured kinematics need review";
  }
  if (workspace.scan_ready && !workspace.workspace?.base_frame_registration_confirmed) {
    return "Scanning ready · confirm base-frame registration";
  }
  if (workspace.scan_ready) return "Scanning ready · confirm kinematics before manipulation";
  if (workspace.intrinsics_ready) return "Camera calibrated · teach travel and survey poses";
  return "Camera calibration required";
}

function kinematicMetricsHtml(validation) {
  if (validation.hand_eye_translation_spread_meters === undefined) return "";
  const metrics = [
    { label: "Camera translation spread", value: validation.hand_eye_translation_spread_meters * 1000, unit: "mm", limit: 25, want: "max" },
    { label: "Camera rotation spread", value: validation.hand_eye_rotation_spread_degrees, unit: "°", limit: 10, want: "max" },
    { label: "Camera mount offset", value: validation.maximum_camera_mount_offset_meters * 1000, unit: "mm", limit: 250, want: "max" },
    { label: "Viewpoint baseline", value: validation.camera_baseline_meters * 1000, unit: "mm", limit: 40, want: "min" },
    { label: "Wrist-frame rotation", value: validation.gripper_rotation_spread_degrees, unit: "°", limit: 15, want: "min" },
    { label: "Joint spread", value: validation.joint_spread_degrees, unit: "°", limit: 15, want: "min" },
  ];
  const rows = metrics.map((metric) => {
    const pass = metric.want === "max" ? metric.value <= metric.limit : metric.value >= metric.limit;
    const shown = metric.unit === "mm" ? metric.value.toFixed(0) : metric.value.toFixed(1);
    const comparator = metric.want === "max" ? "≤" : "≥";
    return `<div class="kin-metric ${pass ? "ok" : "bad"}">
        <i aria-hidden="true">${pass ? "✓" : "✗"}</i>
        <span>${metric.label}</span>
        <b>${shown} ${metric.unit}</b>
        <small>${comparator} ${metric.limit} ${metric.unit}</small>
      </div>`;
  }).join("");
  return `<div class="kin-metrics">${rows}</div>`;
}

function renderPhysicalCalibrationState(state) {
  const workspace = state.robot.workspace || {};
  const samples = workspace.intrinsic_samples || 0;
  const required = workspace.intrinsic_samples_required || 10;
  const intrinsicStatus = $("#intrinsic-calibration-status");
  intrinsicStatus.textContent = workspace.intrinsics_ready
    ? `Calibrated · RMS ${Number(workspace.intrinsic_rms_pixels).toFixed(2)} px`
    : `${samples} of ${required} samples captured`;
  intrinsicStatus.classList.toggle("success", Boolean(workspace.intrinsics_ready));

  $("#travel-pose-state").textContent = workspace.travel_pose_ready
    ? "Recorded"
    : "Not recorded";
  for (const [name, record] of Object.entries(workspace.viewpoints || {})) {
    const element = $(`[data-viewpoint-state="${name}"]`);
    if (!element) continue;
    element.textContent = record.ready
      ? `Recorded · ${Number(record.reprojection_rmse_pixels).toFixed(2)} px`
      : "Not recorded";
  }

  $("#capture-intrinsic-button").disabled = !state.robot.connected || state.robot.autonomous_active;
  $("#solve-intrinsic-button").disabled = samples < required || state.robot.autonomous_active;
  $("#record-travel-button").disabled = !workspace.intrinsics_ready || state.robot.autonomous_active;
  $$('[data-record-viewpoint]').forEach((button) => {
    button.disabled = !workspace.intrinsics_ready || state.robot.autonomous_active;
  });
  $("#save-workspace-button").disabled = !workspace.scan_ready || state.robot.autonomous_active;
  $("#calibration-arm-button").textContent = state.robot.armed
    ? "Disarm manual control"
    : "Arm manual control";

  const validation = workspace.kinematic_validation || {};
  const title = validation.status === "pass"
    ? "Kinematic check passed"
    : validation.status === "review"
      ? "Kinematic check blocked — fix the ✗ rows"
      : "Kinematic check";
  const body = (validation.status === "pass" || validation.status === "review")
    ? kinematicMetricsHtml(validation)
    : `<span>${escapeHtml(validation.reason || "Record three distinct survey poses first.")}</span>`;
  for (const selector of ["#kinematic-validation-status", "#survey-kinematic-status"]) {
    const element = $(selector);
    if (!element) continue;
    element.innerHTML = `<b>${title}</b>${body}`;
    element.classList.toggle("success", validation.status === "pass");
  }

  $("#physical-ready-title").textContent = workspace.execution_ready
    ? "Ready for gated physical testing"
    : workspace.scan_ready
      ? "Ready for scanning"
      : "Calibration incomplete";
  $("#physical-ready-copy").textContent = workspace.execution_ready
    ? "Three-view scans and exact-plan review are enabled. Every scan and every manipulation still requires a fresh approval."
    : workspace.scan_ready
      ? "Three-view scans are enabled. Return to Workspace and complete both base-frame and kinematic confirmations before physical manipulation."
      : "Complete the missing camera or survey-pose steps before moving the arm automatically.";
}

function renderRobotMode(state) {
  const realMode = state.robot.connected;
  const workspace = state.robot.workspace || {};
  const physicalPlan = state.workflow.physical_plan;
  document.body.classList.toggle("real-mode", realMode);
  $("#robot-mode-simulation").classList.toggle("active", !realMode);
  $("#robot-mode-real").classList.toggle("real-active", realMode);
  $("#real-safety-bar").hidden = !realMode;
  $("#main-feed-name").textContent = realMode ? "Digital twin" : "Simulation";
  $("#real-safety-status").textContent = state.robot.autonomous_active
    ? state.robot.status
    : state.robot.armed
      ? "Connected · manual control armed"
      : state.robot.scan_approved
        ? "One survey scan approved"
        : "Controls locked · torque holding pose";
  $("#real-workspace-status").textContent = physicalWorkspaceSummary(workspace);
  $("#arm-real-button").textContent = state.robot.armed ? "Disarm manual control" : "Arm manual control";
  $("#arm-real-button").disabled = state.robot.autonomous_active || state.workflow.busy;
  $("#configure-real-button").disabled = state.robot.autonomous_active || state.workflow.busy;

  const scanButton = $("#approve-scan-button");
  scanButton.textContent = state.robot.scan_approved ? "Scan approved" : "Approve one scan";
  scanButton.disabled = !workspace.scan_ready
    || state.robot.scan_approved
    || state.robot.autonomous_active
    || state.workflow.busy;

  const planReview = $("#physical-plan-review");
  planReview.hidden = !physicalPlan;
  if (physicalPlan) {
    const plannedFrom = physicalPlan.planned_from_positions || {};
    const currentPositions = state.robot.positions || {};
    const movedJoints = Object.entries(plannedFrom).filter(([joint, value]) => {
      const current = Number(currentPositions[joint]);
      const tolerance = joint === "gripper" ? 8 : 2;
      return !Number.isFinite(current) || Math.abs(current - Number(value)) > tolerance;
    }).map(([joint]) => joint.replaceAll("_", " "));
    const startStateChanged = Object.keys(plannedFrom).length === 0 || movedJoints.length > 0;
    if (physicalPlan.plan_id !== lastPhysicalPlanId) {
      $("#physical-plan-confirm").checked = false;
      lastPhysicalPlanId = physicalPlan.plan_id;
    }
    $("#physical-plan-state").textContent = startStateChanged
      ? "PLAN STALE · REPLAN FROM THE CURRENT JOINT STATE"
      : physicalPlan.safe
      ? "PREFLIGHT PASSED · PHYSICAL APPROVAL REQUIRED"
      : "PREFLIGHT BLOCKED · MOTION DISABLED";
    $("#physical-plan-title").textContent = `${physicalPlan.object_label} → ${physicalPlan.destination}`;
    $("#physical-plan-coordinates").textContent = [
      `pick (${physicalPlan.pick_xyz.join(", ")}) m`,
      `place (${physicalPlan.place_xyz.join(", ")}) m`,
      physicalPlan.safe ? "preflight passed" : "preflight blocked",
    ].join(" · ");
    const failedChecks = Object.entries(physicalPlan.checks || {})
      .filter(([, passed]) => !passed)
      .map(([name]) => name.replaceAll("_", " "));
    if (startStateChanged) {
      failedChecks.push(
        movedJoints.length
          ? `arm moved after planning (${movedJoints.join(", ")})`
          : "plan has no recorded starting joint state",
      );
    }
    const notChecked = physicalPlan.not_checked || [];
    $("#physical-plan-checks").innerHTML = [
      failedChecks.length
        ? `<p class="blocked"><b>Blocked:</b> ${escapeHtml(failedChecks.join(", "))}</p>`
        : "<p><b>Checked:</b> base registration, reconstruction diagnostics, IK, calibrated limits, sampled model collisions, scene confidence, and clearances.</p>",
      notChecked.length
        ? `<p><b>Inspect yourself:</b> ${escapeHtml(notChecked.join("; "))}.</p>`
        : "",
    ].join("");
    const steps = physicalPlan.motion_steps || [];
    $("#physical-plan-summary").textContent = `Review ${steps.length} exact joint targets`;
    $("#physical-plan-steps").innerHTML = steps.map((step, index) => `
      <p>
        <b>${index + 1}. ${escapeHtml(step.name)}</b>
        <span>${Object.entries(step.targets).map(([joint, value]) =>
          `${escapeHtml(joint.replaceAll("_", " "))} ${Number(value).toFixed(1)}`
        ).join(" · ")}</span>
      </p>
    `).join("");
    const approveButton = $("#approve-plan-button");
    approveButton.dataset.planId = physicalPlan.plan_id || "";
    approveButton.disabled = !physicalPlan.safe
      || startStateChanged
      || !workspace.execution_ready
      || state.robot.autonomous_active
      || state.workflow.busy
      || !$("#physical-plan-confirm").checked;
  } else {
    lastPhysicalPlanId = null;
    $("#physical-plan-confirm").checked = false;
    $("#approve-plan-button").dataset.planId = "";
  }

  $("#manual-help").textContent = realMode
    ? state.robot.armed
      ? "Physical manual control is armed. Hold a control while watching the arm; release or a lost browser heartbeat stops the motion."
      : "Physical movement is locked. Arm manual control from the safety bar first."
    : "Keyboard pairs are shown on each control. Release stops motion immediately.";
  $$(".hold-button").forEach((button) => {
    button.disabled = realMode && !state.robot.armed;
  });
  $$("[data-manual-action]").forEach((button) => {
    button.disabled = realMode;
  });
  renderPhysicalCalibrationState(state);
  renderTeleopState(state);
  return realMode;
}

function renderState(state) {
  latestState = state;
  const realMode = renderRobotMode(state);
  const readiness = readinessMap(state.readiness);
  const labelsReady = Boolean(readiness.get("05S"));
  renderArtifacts(state.artifacts);
  renderSemanticMemory(state.scene, labelsReady);
  renderReconstructionStages(state.scene.reconstruction);
  renderMemoryActivity(state, readiness);
  renderChat(state, realMode);
  renderHardwareSetup(state.hardware);

  $("#pipeline-log").textContent = state.workflow.logs.join("\n");
  $("#pipeline-log").scrollTop = $("#pipeline-log").scrollHeight;
  const streamFps = state.stream.fps > 0 ? state.stream.fps.toFixed(1) : "—";
  $("#main-stream-meta").textContent = realMode
    ? `${state.stream.main_resolution.join(" × ")} · mirroring physical arm`
    : `${state.stream.main_resolution.join(" × ")} · ${streamFps} FPS`;
  $("#wrist-stream-meta").textContent = `${state.stream.wrist_resolution.join(" × ")} · ${realMode ? "LeRobot" : "native"}`;
  $("#connection-dot").classList.add("connected");
  $("#connection-label").textContent = realMode
    ? state.robot.status
    : state.workflow.busy ? "Working" : "MuJoCo ready";
  $("#clear-memory-button").disabled = state.workflow.busy;

  if (state.workflow.error && state.workflow.error !== lastError) {
    lastError = state.workflow.error;
    showToast(state.workflow.error, true);
  }
  if (!state.workflow.error) lastError = null;
}

async function pollState() {
  try {
    renderState(await api("/api/state"));
  } catch (_) {
    $("#connection-dot").classList.remove("connected");
    $("#connection-label").textContent = "Portal disconnected";
  }
}

async function submitChatMessage(message) {
  if (!message) return;
  const settings = currentLlmSettings();
  const configurationError = llmConfigurationError(settings);
  if (configurationError) {
    populateLlmDialog();
    $("#llm-dialog").showModal();
    showToast(configurationError, true);
    return;
  }
  try {
    await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, settings }),
    });
    await pollState();
    return true;
  } catch (error) {
    showToast(error.message, true);
    return false;
  }
}

async function sendChat() {
  const input = $("#chat-input");
  const message = input.value.trim();
  if (await submitChatMessage(message)) {
    input.value = "";
    input.style.height = "";
  }
}

async function clearChat() {
  try {
    await api("/api/chat/clear", { method: "POST" });
    chatSignature = null;
    showToast("Started a new chat");
    await pollState();
    $("#chat-input").focus();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function clearSceneMemory(includeCalibration) {
  try {
    await api("/api/memory/clear", {
      method: "POST",
      body: JSON.stringify({ include_calibration: includeCalibration }),
    });
    artifactSignature = null;
    memorySignature = null;
    reconstructionSignature = null;
    $("#memory-dialog").close();
    showToast(includeCalibration ? "All scene memory cleared" : "Object memory cleared");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function connectRealRobot() {
  for (let step = 1; step <= 4; step += 1) {
    const error = robotSetupValidation(step);
    if (error) {
      showRobotSetupStep(step, error);
      return;
    }
  }

  const button = $("#connect-real-button");
  button.disabled = true;
  button.textContent = "Connecting…";
  try {
    await api("/api/robot/connect", {
      method: "POST",
      body: JSON.stringify({
        port: $("#real-port").value.trim(),
        robot_id: $("#real-robot-id").value.trim(),
        camera_index: Number($("#real-camera-index").value),
        width: Number($("#real-camera-width").value),
        height: Number($("#real-camera-height").value),
        fps: Number($("#real-camera-fps").value),
        rotation: Number($("#real-camera-rotation").value),
      }),
    });
    saveHardwarePreferences();
    $("#robot-dialog").close();
    showToast("Real SO-101 connected · controls locked · torque holding pose");
    await pollState();
  } catch (error) {
    setRobotSetupMessage(error.message);
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Connect safely";
  }
}

async function discoverHardware() {
  const button = $("#discover-hardware-button");
  button.disabled = true;
  button.textContent = "Scanning…";
  $("#hardware-discovery-status").textContent = "Scanning serial devices and cameras";
  setRobotSetupMessage("Checking LeRobot, serial devices, cameras, and saved calibration…", false);
  try {
    const hardware = await api("/api/hardware/discover", {
      method: "POST",
      body: JSON.stringify({ include_cameras: true }),
    });
    hardwareSignature = null;
    renderHardwareSetup(hardware);
    robotSetupScanComplete = true;
    const error = robotSetupValidation(2);
    setRobotSetupMessage(
      error || "Hardware scan complete. Confirm the follower and wrist-camera assignments.",
      Boolean(error),
    );
  } catch (error) {
    setRobotSetupMessage(error.message);
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Scan connected hardware";
  }
}

function saveHardwarePreferences() {
  localStorage.setItem(hardwarePreferencesKey, JSON.stringify({
    followerPort: $("#real-port").value.trim(),
    followerId: $("#real-robot-id").value.trim(),
    leaderPort: $("#real-leader-port").value.trim(),
    leaderId: $("#real-leader-id").value.trim(),
    cameraIndex: $("#real-camera-index").value,
    cameraWidth: $("#real-camera-width").value,
    cameraHeight: $("#real-camera-height").value,
    cameraFps: $("#real-camera-fps").value,
    cameraRotation: $("#real-camera-rotation").value,
    rotationConvention: "counter-clockwise-v1",
  }));
}

function restoreHardwarePreferences() {
  const preferences = storedJson(hardwarePreferencesKey);
  if (
    preferences.cameraRotation !== undefined
    && preferences.rotationConvention !== "counter-clockwise-v1"
  ) {
    if (String(preferences.cameraRotation) === "90") preferences.cameraRotation = "270";
    else if (String(preferences.cameraRotation) === "270") preferences.cameraRotation = "90";
    preferences.rotationConvention = "counter-clockwise-v1";
    localStorage.setItem(hardwarePreferencesKey, JSON.stringify(preferences));
  }
  const fields = {
    followerPort: "#real-port",
    followerId: "#real-robot-id",
    leaderPort: "#real-leader-port",
    leaderId: "#real-leader-id",
    cameraIndex: "#real-camera-index",
    cameraWidth: "#real-camera-width",
    cameraHeight: "#real-camera-height",
    cameraFps: "#real-camera-fps",
    cameraRotation: "#real-camera-rotation",
  };
  for (const [name, selector] of Object.entries(fields)) {
    if (preferences[name] !== undefined) $(selector).value = preferences[name];
  }
}

async function disconnectRealRobot() {
  await releaseAll();
  try {
    await api("/api/robot/disconnect", { method: "POST" });
    showToast("Physical robot disconnected · MuJoCo resumed");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function toggleRealArm() {
  const armed = !latestState?.robot.armed;
  try {
    await api("/api/robot/arm", {
      method: "POST",
      body: JSON.stringify({ armed }),
    });
    showToast(armed ? "Physical manual control armed" : "Physical manual control disarmed");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

const RECORDING_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function formatRecordingLabel(recording) {
  const match = recording.name.match(/(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
  const when = match
    ? `${RECORDING_MONTHS[Number(match[2]) - 1]} ${Number(match[3])}, ${match[4]}:${match[5]}`
    : recording.name;
  const seconds = `${Number(recording.duration_seconds).toFixed(1)}s`;
  return `Take ${recording.take} · ${when} · ${seconds} · ${recording.frames}f`;
}

async function fetchTeleopRecordings() {
  try {
    const data = await api("/api/robot/teleop/recordings");
    const select = $("#teleop-recording-select");
    const previous = select.value;
    const raw = data.recordings || [];
    const recordings = raw.map((recording, index) => ({ ...recording, take: raw.length - index }));
    select.innerHTML = recordings
      .map((recording) =>
        `<option value="${escapeHtml(recording.name)}">${escapeHtml(formatRecordingLabel(recording))}</option>`,
      )
      .join("");
    if (recordings.some((recording) => recording.name === previous)) {
      select.value = previous;
    } else if (recordings.length) {
      select.selectedIndex = 0;
    }
    $("#teleop-recordings-empty").hidden = recordings.length > 0;
    $("#teleop-recordings-count").textContent = recordings.length ? `(${recordings.length})` : "";
    if (latestState) renderTeleopState(latestState);
  } catch (_) {}
}

function openTeleopPanel() {
  const panel = $("#teleop-panel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) {
    const preferences = storedJson(hardwarePreferencesKey);
    if (!$("#teleop-leader-port").value) {
      $("#teleop-leader-port").value = $("#real-leader-port")?.value?.trim() || preferences.leaderPort || "";
    }
    if (!$("#teleop-leader-id").value) {
      $("#teleop-leader-id").value = $("#real-leader-id")?.value?.trim() || preferences.leaderId || "";
    }
    fetchTeleopRecordings();
  }
}

async function toggleTeleop() {
  if (latestState?.robot?.teleop_active) {
    try {
      const result = await api("/api/robot/teleop/stop", { method: "POST" });
      showToast(result.saved?.saved ? `Teleop stopped · saved ${result.saved.frames}-frame recording` : "Teleop stopped");
      await fetchTeleopRecordings();
      await pollState();
    } catch (error) {
      showToast(error.message, true);
    }
    return;
  }
  const leaderPort = $("#teleop-leader-port").value.trim();
  const leaderId = $("#teleop-leader-id").value.trim();
  if (!leaderPort || !leaderId) {
    showToast("Enter the leader USB port and calibration ID", true);
    return;
  }
  try {
    await api("/api/robot/teleop/start", {
      method: "POST",
      body: JSON.stringify({ leader_port: leaderPort, leader_id: leaderId }),
    });
    showToast("Teleop started — move the leader arm to drive the follower");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function toggleTeleopRecording() {
  try {
    if (latestState?.robot?.recording) {
      const result = await api("/api/robot/teleop/record/stop", { method: "POST" });
      showToast(result.saved ? `Recorded ${result.frames} frames` : result.reason || "Recording too short");
      await fetchTeleopRecordings();
    } else {
      await api("/api/robot/teleop/record/start", { method: "POST" });
      showToast("Recording — move the leader arm, then press Stop recording");
    }
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function replayTeleop() {
  const name = $("#teleop-recording-select").value;
  if (!name) {
    showToast("Select a recording to replay", true);
    return;
  }
  if (!window.confirm("Replay this trajectory on the real arm at recorded speed? Keep Stop + disconnect within reach.")) return;
  const button = $("#teleop-replay-button");
  button.disabled = true;
  button.textContent = "Replaying…";
  try {
    await api("/api/robot/teleop/replay", { method: "POST", body: JSON.stringify({ name }) });
    showToast("Replay complete");
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.textContent = "▶ Replay";
  }
}

async function deleteTeleopRecording() {
  const select = $("#teleop-recording-select");
  const name = select.value;
  if (!name) {
    showToast("Select a recording to delete", true);
    return;
  }
  const label = select.options[select.selectedIndex]?.textContent || name;
  if (!window.confirm(`Delete "${label}"? This cannot be undone.`)) return;
  try {
    await api(`/api/robot/teleop/recordings/${encodeURIComponent(name)}`, { method: "DELETE" });
    showToast("Recording deleted");
    await fetchTeleopRecordings();
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderTeleopState(state) {
  const active = state.robot.teleop_active;
  const recording = state.robot.recording;
  const busy = state.robot.autonomous_active || state.workflow.busy;
  $("#teleop-toggle-button").classList.toggle("real-active", active);
  const startButton = $("#teleop-start-button");
  startButton.textContent = active ? "Stop teleop" : "Start teleop";
  startButton.classList.toggle("emergency", active);
  startButton.disabled = busy;
  const portField = $("#teleop-leader-port");
  const idField = $("#teleop-leader-id");
  portField.disabled = active;
  idField.disabled = active;
  if (!active) {
    const preferences = storedJson(hardwarePreferencesKey);
    if (!portField.value) portField.value = $("#real-leader-port")?.value?.trim() || preferences.leaderPort || "";
    if (!idField.value) idField.value = $("#real-leader-id")?.value?.trim() || preferences.leaderId || "";
  }
  const recordButton = $("#teleop-record-button");
  recordButton.hidden = !active;
  recordButton.textContent = recording ? "■ Stop recording" : "● Record";
  recordButton.classList.toggle("danger-ghost", recording);
  const hasSelection = Boolean($("#teleop-recording-select").value);
  const replayButton = $("#teleop-replay-button");
  replayButton.disabled = active || busy || !hasSelection;
  $("#teleop-delete-button").disabled = active || busy || !hasSelection;
  const status = $("#teleop-status");
  if (recording) {
    status.textContent = `Recording · ${state.robot.recorded_frames} frames captured. Move the leader; press Stop recording when done.`;
  } else if (active) {
    status.textContent = "Teleop live · the follower mirrors your leader arm. Press Record to capture a trajectory.";
  } else {
    status.textContent = "Move the leader arm to drive the follower. Start teleop, then optionally record and replay.";
  }
}

function setPhysicalCalibrationMessage(message = "", isError = true) {
  const element = $("#physical-calibration-message");
  element.textContent = message;
  element.hidden = !message;
  element.className = `robot-setup-message${isError ? "" : " info"}`;
}

function populatePhysicalWorkspaceFields() {
  const workspace = latestState?.robot.workspace?.workspace;
  if (!workspace) return;
  const values = {
    "#workspace-x-min": workspace.minimum?.[0],
    "#workspace-x-max": workspace.maximum?.[0],
    "#workspace-y-min": workspace.minimum?.[1],
    "#workspace-y-max": workspace.maximum?.[1],
    "#workspace-z-max": workspace.maximum?.[2],
    "#table-x-min": workspace.table_minimum?.[0],
    "#table-x-max": workspace.table_maximum?.[0],
    "#table-y-min": workspace.table_minimum?.[1],
    "#table-y-max": workspace.table_maximum?.[1],
    "#destination-left-y": workspace.destinations?.["left side"]?.[1],
    "#destination-x": workspace.destinations?.center?.[0],
    "#destination-right-y": workspace.destinations?.["right side"]?.[1],
    "#gripper-open": workspace.gripper_open_percent,
    "#gripper-closed": workspace.gripper_closed_percent,
  };
  for (const [selector, value] of Object.entries(values)) {
    if (value !== undefined) $(selector).value = value;
  }
  $("#confirm-kinematic-model").checked = Boolean(workspace.kinematic_model_confirmed);
  $("#confirm-base-registration").checked = Boolean(workspace.base_frame_registration_confirmed);
}

function showPhysicalCalibrationStep(step, message = "") {
  physicalCalibrationStep = Math.max(1, Math.min(4, step));
  const labels = ["", "Camera", "Survey poses", "Workspace", "Ready"];
  $$('[data-physical-step]').forEach((section) => {
    section.hidden = Number(section.dataset.physicalStep) !== physicalCalibrationStep;
  });
  $$('[data-physical-progress]').forEach((item) => {
    const itemStep = Number(item.dataset.physicalProgress);
    item.classList.toggle("current", itemStep === physicalCalibrationStep);
    item.classList.toggle("complete", itemStep < physicalCalibrationStep);
    if (itemStep === physicalCalibrationStep) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  });
  $("#physical-calibration-position").textContent = `Step ${physicalCalibrationStep} of 4 · ${labels[physicalCalibrationStep]}`;
  $("#physical-calibration-back").hidden = physicalCalibrationStep === 1;
  $("#physical-calibration-next").hidden = physicalCalibrationStep === 4;
  $(".physical-calibration-content").scrollTop = 0;
  setPhysicalCalibrationMessage(message);
}

function openPhysicalCalibration() {
  const workspace = latestState?.robot.workspace || {};
  if (!latestState?.robot.connected) {
    showToast("Connect the physical SO-101 first", true);
    return;
  }
  populatePhysicalWorkspaceFields();
  const viewpointsReady = Object.values(workspace.viewpoints || {}).every((view) => view.ready);
  const step = !workspace.intrinsics_ready
    ? 1
    : !workspace.travel_pose_ready
      || !viewpointsReady
      || workspace.kinematic_validation?.status !== "pass"
      ? 2
      : workspace.execution_ready
        ? 4
        : 3;
  showPhysicalCalibrationStep(step);
  $("#physical-calibration-dialog").showModal();
}

function physicalCalibrationValidation(step) {
  const workspace = latestState?.robot.workspace || {};
  if (step === 1 && !workspace.intrinsics_ready) {
    return "Capture at least ten varied checkerboard views, then solve the camera intrinsics.";
  }
  if (step === 2) {
    const missing = Object.entries(workspace.viewpoints || {})
      .filter(([, record]) => !record.ready)
      .map(([name]) => name.replace("survey_", ""));
    if (!workspace.travel_pose_ready) missing.unshift("safe travel");
    if (missing.length) return `Record: ${missing.join(", ")}.`;
    if (workspace.kinematic_validation?.status !== "pass") {
      return "The three poses do not yet form a consistent kinematic check. Re-record distinct left, center, and right views.";
    }
  }
  if (step === 3) {
    if (!workspace.workspace?.base_frame_registration_confirmed) {
      return "Confirm that every measurement uses the robot base datum, then save the workspace.";
    }
    if (!workspace.workspace?.kinematic_model_confirmed) {
      return "Confirm the physical zero pose and all five joint directions, then save the workspace.";
    }
    if (!workspace.execution_ready) {
      return "Save the workspace and resolve the measured kinematic check before continuing.";
    }
  }
  return null;
}

function advancePhysicalCalibration() {
  const error = physicalCalibrationValidation(physicalCalibrationStep);
  if (error) {
    setPhysicalCalibrationMessage(error);
    return;
  }
  showPhysicalCalibrationStep(physicalCalibrationStep + 1);
}

async function runCalibrationRequest(button, path, body, successMessage) {
  button.disabled = true;
  try {
    const result = await api(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    setPhysicalCalibrationMessage(successMessage(result), false);
    await pollState();
  } catch (error) {
    setPhysicalCalibrationMessage(error.message);
  } finally {
    button.disabled = false;
  }
}

async function savePhysicalWorkspace() {
  const number = (selector) => {
    const input = $(selector);
    return input.value.trim() === "" ? Number.NaN : Number(input.value);
  };
  const destinationX = number("#destination-x");
  const payload = {
    minimum: [number("#workspace-x-min"), number("#workspace-y-min"), 0.004],
    maximum: [number("#workspace-x-max"), number("#workspace-y-max"), number("#workspace-z-max")],
    table_minimum: [number("#table-x-min"), number("#table-y-min")],
    table_maximum: [number("#table-x-max"), number("#table-y-max")],
    destinations: {
      "left side": [destinationX, number("#destination-left-y")],
      center: [destinationX, 0],
      "right side": [destinationX, number("#destination-right-y")],
    },
    gripper_open_percent: number("#gripper-open"),
    gripper_closed_percent: number("#gripper-closed"),
    base_frame_registration_confirmed: $("#confirm-base-registration").checked,
    kinematic_model_confirmed: $("#confirm-kinematic-model").checked,
  };
  const numericValues = [
    ...payload.minimum,
    ...payload.maximum,
    ...payload.table_minimum,
    ...payload.table_maximum,
    ...Object.values(payload.destinations).flat(),
    payload.gripper_open_percent,
    payload.gripper_closed_percent,
  ];
  if (numericValues.some((value) => !Number.isFinite(value))) {
    setPhysicalCalibrationMessage("Complete every workspace measurement before saving.");
    return;
  }
  await runCalibrationRequest(
    $("#save-workspace-button"),
    "/api/robot/calibration/workspace",
    payload,
    () => payload.base_frame_registration_confirmed && payload.kinematic_model_confirmed
      ? "Workspace saved. Manipulation is enabled only if the measured kinematic check also passes."
      : "Workspace saved. Scanning is enabled; manipulation stays locked until both confirmations are complete.",
  );
}

async function approvePhysicalScan() {
  if (!window.confirm(
    "Approve one physical survey? The arm will move slowly through the saved travel, left, center, and right camera poses. Keep Stop + disconnect within reach.",
  )) return;
  try {
    await api("/api/robot/approve-scan", {
      method: "POST",
      body: JSON.stringify({ confirmation: "APPROVE ONE SCAN" }),
    });
    showToast("One three-view scan approved");
    await pollState();
    if (latestState?.chat.pending_action) {
      await submitChatMessage("Continue");
    }
  } catch (error) {
    showToast(error.message, true);
  }
}

async function approvePhysicalPlan() {
  const planId = $("#approve-plan-button").dataset.planId;
  if (!planId || !$("#physical-plan-confirm").checked) return;
  try {
    await api("/api/robot/approve-plan", {
      method: "POST",
      body: JSON.stringify({
        plan_id: planId,
        confirmation: "APPROVE PHYSICAL MOTION",
      }),
    });
    showToast("Exact physical plan approved · keep Stop within reach");
    $("#physical-plan-confirm").checked = false;
    await pollState();
  } catch (error) {
    showToast(error.message, true);
  }
}

function bindActions() {
  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    sendChat();
  });
  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  });
  $("#chat-input").addEventListener("input", (event) => {
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 120)}px`;
  });
  $$("[data-manual-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await api(`/api/manual/${button.dataset.manualAction}`, { method: "POST" });
        showToast(`${button.textContent.trim()} requested`);
      } catch (error) {
        showToast(error.message, true);
      }
    });
  });
  $("#clear-memory-button").addEventListener("click", () => $("#memory-dialog").showModal());
  $("#new-chat-button").addEventListener("click", clearChat);
  $("#chat-model").addEventListener("click", () => {
    populateLlmDialog();
    $("#llm-dialog").showModal();
  });
  $("#llm-provider").addEventListener("change", (event) => changeLlmProvider(event.target.value));
  $("#toggle-llm-key").addEventListener("click", () => {
    const input = $("#llm-api-key");
    input.type = input.type === "password" ? "text" : "password";
    $("#toggle-llm-key").textContent = input.type === "password" ? "Show" : "Hide";
  });
  $("#save-llm-settings").addEventListener("click", saveLlmSettings);
  $("#forget-objects-button").addEventListener("click", () => clearSceneMemory(false));
  $("#clear-all-memory-button").addEventListener("click", () => clearSceneMemory(true));
  $("#copy-log").addEventListener("click", async () => {
    await navigator.clipboard.writeText($("#pipeline-log").textContent);
    showToast("Pipeline log copied");
  });
  $("#robot-mode-real").addEventListener("click", () => {
    if (latestState?.robot.connected) {
      showToast("The physical robot is connected; use the safety bar to disconnect.");
    } else {
      openRobotSetup();
    }
  });
  $("#robot-mode-simulation").addEventListener("click", () => {
    if (latestState?.robot.connected) disconnectRealRobot();
  });
  $("#robot-setup-next").addEventListener("click", advanceRobotSetup);
  $("#robot-setup-back").addEventListener("click", returnToPreviousRobotSetupStep);
  $("#connect-real-button").addEventListener("click", connectRealRobot);
  $("#discover-hardware-button").addEventListener("click", discoverHardware);
  ["#real-robot-id", "#real-leader-id"].forEach((selector) => {
    $(selector).addEventListener("input", () => {
      renderSelectedCalibration();
      setRobotSetupMessage();
    });
  });
  [
    "#real-port",
    "#real-leader-port",
    "#real-camera-index",
    "#real-camera-width",
    "#real-camera-height",
    "#real-camera-fps",
    "#real-camera-rotation",
  ].forEach((selector) => {
    $(selector).addEventListener("input", () => {
      renderRobotSetupReview();
      setRobotSetupMessage();
    });
    $(selector).addEventListener("change", () => {
      renderRobotSetupReview();
      setRobotSetupMessage();
    });
  });
  [
    "#setup-confirm-built",
    "#setup-confirm-secure",
    "#setup-confirm-stop",
    "#setup-confirm-connect",
  ].forEach((selector) => {
    $(selector).addEventListener("change", () => setRobotSetupMessage());
  });
  $("#arm-real-button").addEventListener("click", toggleRealArm);
  $("#teleop-toggle-button").addEventListener("click", openTeleopPanel);
  $("#teleop-start-button").addEventListener("click", toggleTeleop);
  $("#teleop-record-button").addEventListener("click", toggleTeleopRecording);
  $("#teleop-replay-button").addEventListener("click", replayTeleop);
  $("#teleop-delete-button").addEventListener("click", deleteTeleopRecording);
  $("#teleop-recordings-refresh").addEventListener("click", fetchTeleopRecordings);
  $("#teleop-recording-select").addEventListener("change", () => {
    if (latestState) renderTeleopState(latestState);
  });
  $("#teleop-recording-select").addEventListener("dblclick", replayTeleop);
  $("#disconnect-real-button").addEventListener("click", disconnectRealRobot);
  $("#configure-real-button").addEventListener("click", openPhysicalCalibration);
  $("#approve-scan-button").addEventListener("click", approvePhysicalScan);
  $("#approve-plan-button").addEventListener("click", approvePhysicalPlan);
  $("#physical-plan-confirm").addEventListener("change", () => {
    if (latestState) renderRobotMode(latestState);
  });
  $("#physical-calibration-next").addEventListener("click", advancePhysicalCalibration);
  $("#physical-calibration-back").addEventListener("click", () => {
    showPhysicalCalibrationStep(physicalCalibrationStep - 1);
  });
  $("#calibration-arm-button").addEventListener("click", toggleRealArm);
  $("#capture-intrinsic-button").addEventListener("click", (event) => {
    runCalibrationRequest(
      event.currentTarget,
      "/api/robot/calibration/intrinsics/sample",
      undefined,
      (result) => `Checkerboard sample ${result.samples} captured. Change its angle and distance before the next sample.`,
    );
  });
  $("#solve-intrinsic-button").addEventListener("click", (event) => {
    runCalibrationRequest(
      event.currentTarget,
      "/api/robot/calibration/intrinsics/solve",
      undefined,
      (result) => `Camera intrinsics saved · ${Number(result.rms_pixels).toFixed(2)} px RMS.`,
    );
  });
  $("#record-travel-button").addEventListener("click", (event) => {
    runCalibrationRequest(
      event.currentTarget,
      "/api/robot/calibration/travel",
      undefined,
      () => "Safe travel pose recorded from the current physical joints.",
    );
  });
  $$('[data-record-viewpoint]').forEach((button) => {
    button.addEventListener("click", (event) => {
      const boardPose = {
        name: event.currentTarget.dataset.recordViewpoint,
        board_origin_x: Number($("#board-origin-x").value),
        board_origin_y: Number($("#board-origin-y").value),
        board_origin_z: Number($("#board-origin-z").value),
        board_yaw_degrees: Number($("#board-yaw").value),
      };
      const boardInputs = [
        "#board-origin-x",
        "#board-origin-y",
        "#board-origin-z",
        "#board-yaw",
      ];
      if (boardInputs.some((selector) => $(selector).value.trim() === "")
          || Object.values(boardPose).slice(1).some((value) => !Number.isFinite(value))) {
        setPhysicalCalibrationMessage("Enter all four measured checkerboard pose values first.");
        return;
      }
      runCalibrationRequest(
        event.currentTarget,
        "/api/robot/calibration/viewpoint",
        boardPose,
        (result) => `${result.viewpoint.replace("survey_", "")} view recorded · ${Number(result.reprojection_rmse_pixels).toFixed(2)} px RMS.`,
      );
    });
  });
  $("#save-workspace-button").addEventListener("click", savePhysicalWorkspace);
  $("#clear-physical-calibration-button").addEventListener("click", async () => {
    if (!window.confirm("Delete this camera and workspace profile? This cannot be undone.")) return;
    try {
      await api("/api/robot/calibration", { method: "DELETE" });
      showPhysicalCalibrationStep(1, "Workspace profile deleted. Start with camera calibration.");
      await pollState();
    } catch (error) {
      setPhysicalCalibrationMessage(error.message);
    }
  });
}

buildJointControls();
restoreHardwarePreferences();
bindKeyboard();
bindActions();
renderLlmBadge();
pollState();
connectStream($("#main-stream"), "main");
connectStream($("#wrist-stream"), "wrist");
connectStream($("#calibration-wrist-stream"), "wrist");
connectStream($("#survey-wrist-stream"), "wrist");
setInterval(pollState, 400);
setInterval(() => {
  if (!latestState?.robot.connected) return;
  if (heldKeys.size === 0 && !$(".hold-button.active")) return;
  api("/api/manual/heartbeat", { method: "POST" }).catch(() => {});
}, 100);
