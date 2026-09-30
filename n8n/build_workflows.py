"""Generate the importable n8n workflow JSON in n8n/workflows/.

Edit the workflows here (not in the JSON) and re-run: python n8n/build_workflows.py
Node IDs are deterministic, so regenerating only changes what you actually edited.
"""
import itertools
import json
import os
import uuid
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")
API = "={{ $env.CHURN_API_URL }}"
AUTH = {"sendHeaders": True, "headerParameters": {"parameters": [{"name": "X-API-Key", "value": "={{ $env.CHURN_API_KEY }}"}]}}

_ids = itertools.count()


def nid():
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"churn-platform-n8n/{next(_ids)}"))

def node(name, type_, ver, pos, params, **extra):
    return {"parameters": params, "id": nid(), "name": name, "type": f"n8n-nodes-base.{type_}",
            "typeVersion": ver, "position": pos, **extra}

def http(name, pos, path, method="GET", body=None, url=None, auth=True, **extra):
    p = {"url": url or f"{API}{path}"}
    if method != "GET": p["method"] = method
    if auth: p.update(AUTH)
    if body is not None:
        p.update({"sendBody": True, "specifyBody": "json", "jsonBody": body})
    p["options"] = extra.pop("options", {})
    return node(name, "httpRequest", 4.2, pos, p, **extra)

def code(name, pos, js, mode=None):
    p = {"jsCode": js.strip()}
    if mode: p["mode"] = mode
    return node(name, "code", 2, pos, p)

def schedule(name, pos, cron):
    return node(name, "scheduleTrigger", 1.2, pos, {"rule": {"interval": [{"field": "cronExpression", "expression": cron}]}})

def manual(pos): return node("Run manually", "manualTrigger", 1, pos, {})

def cond(left, op_type, operation, right=None):
    c = {"id": nid(), "leftValue": left, "operator": {"type": op_type, "operation": operation}}
    if right is None: c["operator"]["singleValue"] = True
    else: c["rightValue"] = right
    return {"options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
            "conditions": [c], "combinator": "and"}

def connect(*pairs):
    conns = {}
    for src, dst, *out in pairs:
        idx = out[0] if out else 0
        main = conns.setdefault(src, {"main": []})["main"]
        while len(main) <= idx: main.append([])
        main[idx].append({"node": dst, "type": "main", "index": 0})
    return conns

def workflow(name, wid, nodes, conns, tags):
    return {"id": wid, "name": name, "active": False, "nodes": nodes, "connections": conns,
            "settings": {"executionOrder": "v1"}, "pinData": {}, "meta": {"templateCredsSetupCompleted": True},
            "tags": []}

SLACK_SEND = lambda pos: http("Post to Slack", pos, None, method="POST", url="={{ $env.SLACK_WEBHOOK_URL }}",
                             auth=False, body="={{ JSON.stringify({ text: $json.text }) }}",
                             onError="continueRegularOutput")

# ---------------------------------------------------------------- 1. retention
PLAYBOOK_JS = r"""
// Retention playbook: the business rules live here (editable without redeploying the model).
// churn_probability is calibrated (isotonic, out-of-fold): 0.6 means ~60% of such customers churn.
// On the Telco base (26.5% churn) HIGH flags ~13% of customers, MEDIUM another ~25%.
const HIGH = 0.6;
const MEDIUM = 0.35;

const customers = Object.fromEntries(
  $('Pull active customers').first().json.customers.map(c => [c.customerID, c])
);

function pickOffer(c) {
  if (c.Contract === 'Month-to-month') return '15% off for switching to a 1-year contract';
  if (c.TechSupport === 'No' && c.InternetService !== 'No') return '3 months of free tech support';
  if (c.PaymentMethod === 'Electronic check') return '$5/month credit for enabling autopay';
  return 'loyalty check-in, no discount';
}

const out = [];
for (const item of $input.all()) {
  for (const p of item.json.predictions) {
    const c = customers[p.customer_id] || {};
    const tier = p.churn_probability >= HIGH ? 'high' : p.churn_probability >= MEDIUM ? 'medium' : 'low';
    out.push({ json: {
      customer_id: p.customer_id,
      churn_probability: p.churn_probability,
      tier,
      action: { high: 'retention_call', medium: 'retention_email', low: 'none' }[tier],
      offer: tier === 'low' ? null : pickOffer(c),
      monthly_charges: c.MonthlyCharges,
      tenure_months: c.tenure,
      drivers: p.top_risk_factors.map(f => `${f.feature} = ${f.value}`),
    }});
  }
}
return out;
"""

DIGEST_JS = r"""
// One Slack message per run instead of one per customer. Skipped if SLACK_WEBHOOK_URL is unset.
if (!$env.SLACK_WEBHOOK_URL) return [];

const rows = $input.all().map(i => i.json);
const high = rows.filter(r => r.tier === 'high').sort((a, b) => b.churn_probability - a.churn_probability);
const medium = rows.filter(r => r.tier === 'medium');
const revenueAtRisk = high.concat(medium).reduce((s, r) => s + (r.monthly_charges || 0), 0);

const lines = high.slice(0, 5).map(r =>
  `• *${r.customer_id}* - ${(r.churn_probability * 100).toFixed(0)}% - ${r.drivers.join(', ')} → _${r.offer}_`
);
const text = [
  `*Churn scan:* ${rows.length} customers scored - ${high.length} high risk (call), ${medium.length} medium (email).`,
  `Monthly revenue at risk: $${revenueAtRisk.toFixed(2)}`,
  ...(lines.length ? ['*Top high-risk:*', ...lines] : []),
].join('\n');
return [{ json: { text } }];
"""

ACTION_BODY = ("={{ JSON.stringify({ customer_id: $json.customer_id, action: $json.action, "
               "details: { churn_probability: $json.churn_probability, offer: $json.offer, drivers: $json.drivers } }) }}")

switch = node("Route by risk tier", "switch", 3.2, [1340, 300], {
    "rules": {"values": [
        {"conditions": cond("={{ $json.tier }}", "string", "equals", "high"), "renameOutput": True, "outputKey": "high"},
        {"conditions": cond("={{ $json.tier }}", "string", "equals", "medium"), "renameOutput": True, "outputKey": "medium"},
    ]},
    "options": {},
})
nodes = [
    schedule("Every weekday 08:00", [240, 200], "0 8 * * 1-5"),
    manual([240, 400]),
    http("Pull active customers", [480, 300], "/demo/customers?n=50"),
    http("Score churn risk", [720, 300], "/score", method="POST",
         body="={{ JSON.stringify({ customers: $json.customers }) }}"),
    code("Apply retention playbook", [960, 300], PLAYBOOK_JS),
    switch,
    http("Create call task", [1600, 200], "/actions", method="POST", body=ACTION_BODY),
    http("Queue retention email", [1600, 400], "/actions", method="POST", body=ACTION_BODY),
    code("Build Slack digest", [1340, 560], DIGEST_JS),
    SLACK_SEND([1600, 560]),
]
nodes[5]["position"] = [1340, 300]
conns = connect(
    ("Every weekday 08:00", "Pull active customers"), ("Run manually", "Pull active customers"),
    ("Pull active customers", "Score churn risk"), ("Score churn risk", "Apply retention playbook"),
    ("Apply retention playbook", "Route by risk tier"), ("Apply retention playbook", "Build Slack digest"),
    ("Route by risk tier", "Create call task", 0), ("Route by risk tier", "Queue retention email", 1),
    ("Build Slack digest", "Post to Slack"),
)
wf1 = workflow("Churn - daily retention scoring", "ChurnRetention01", nodes, conns, [])

# ---------------------------------------------------------------- 2. complaints
ROUTE_JS = r"""
// Map the model's category to an owning team; send uncertain or legally sensitive cases to a human.
const TEAMS = {
  credit_card: 'cards-support',
  credit_reporting: 'credit-bureau-disputes',
  debt_collection: 'collections-compliance',
  mortgages_and_loans: 'lending-servicing',
  retail_banking: 'branch-banking',
};
const MIN_CONFIDENCE = 0.6;
const URGENT = /\b(lawyer|attorney|lawsuit|sue|cfpb|fraud|identity theft|stolen)\b/i;

const complaint = $('Complaint received').item.json.body;
const r = $input.item.json;
const lowConfidence = r.confidence < MIN_CONFIDENCE;

return { json: {
  customer_id: complaint.customer_id || null,
  action: 'route_complaint',
  details: {
    category: r.category,
    confidence: r.confidence,
    team: lowConfidence ? 'manual-triage' : TEAMS[r.category],
    priority: URGENT.test(complaint.text) ? 'urgent' : 'normal',
    reason: lowConfidence ? `low model confidence (${r.confidence})` : 'auto-routed',
    excerpt: complaint.text.slice(0, 280),
  },
}};
"""
nodes = [
    node("Complaint received", "webhook", 2, [240, 300],
         {"httpMethod": "POST", "path": "complaint", "responseMode": "lastNode", "options": {}},
         webhookId=nid()),
    http("Classify complaint", [480, 300], "/classify", method="POST",
         body="={{ JSON.stringify({ text: $json.body.text }) }}"),
    code("Route to team", [720, 300], ROUTE_JS, mode="runOnceForEachItem"),
    http("Log routed ticket", [960, 300], "/actions", method="POST", body="={{ JSON.stringify($json) }}"),
]
conns = connect(("Complaint received", "Classify complaint"), ("Classify complaint", "Route to team"),
                ("Route to team", "Log routed ticket"))
wf2 = workflow("Churn - complaint routing", "ChurnComplaint02", nodes, conns, [])

# ---------------------------------------------------------------- 3. drift
RETRAIN_MSG_JS = r"""
const d = $('Check drift').first().json;
const r = $input.first().json;
const verdict = r.promoted ? 'new model promoted' : 'candidate REJECTED (CV ROC-AUC regressed) - old model kept';
const text = [
  `*Drift check: retrain triggered* - ${verdict}.`,
  `Drifted features: ${d.drifted_features.join(', ')} - ` +
    `${(d.drifted_importance_mass * 100).toFixed(0)}% of model importance (threshold ${(d.importance_drift_threshold * 100).toFixed(0)}%).`,
  d.roc_auc_drop === null ? 'No labels supplied - performance check skipped.'
    : `ROC-AUC drop ${d.roc_auc_drop.toFixed(3)} (threshold ${d.roc_auc_drop_threshold}).`,
  `CV ROC-AUC: ${r.previous_cv_roc_auc.toFixed(4)} → ${r.candidate_cv_roc_auc.toFixed(4)}`,
].join('\n');
return [{ json: { text, promoted: r.promoted, previous_cv_roc_auc: r.previous_cv_roc_auc, candidate_cv_roc_auc: r.candidate_cv_roc_auc } }];
"""
SLACK_GATE_JS = "// skip Slack when no webhook is configured\nreturn $env.SLACK_WEBHOOK_URL ? $input.all() : [];"

iff = node("Retrain recommended?", "if", 2, [960, 300],
           {"conditions": cond("={{ $json.retrain_recommended }}", "boolean", "true"), "options": {}})
nodes = [
    schedule("Every Monday 07:00", [240, 200], "0 7 * * 1"),
    manual([240, 400]),
    # swap for a real CRM export; scenario=both injects the simulated price hike + contract shift
    http("Pull current customers", [480, 300], "/demo/customers?n=1000&scenario=both"),
    http("Check drift", [720, 300], "/drift/check", method="POST",
         body="={{ JSON.stringify({ customers: $json.customers }) }}"),
    iff,
    http("Retrain model", [1200, 200], "/retrain", method="POST", options={"timeout": 300000}),
    code("Summarize retrain", [1440, 200], RETRAIN_MSG_JS),
    http("Log retrain", [1680, 100], "/actions", method="POST",
         body="={{ JSON.stringify({ action: $json.promoted ? 'model_promoted' : 'model_rejected', "
              "details: { previous_cv_roc_auc: $json.previous_cv_roc_auc, candidate_cv_roc_auc: $json.candidate_cv_roc_auc } }) }}"),
    code("Slack configured?", [1680, 300], SLACK_GATE_JS),
    SLACK_SEND([1920, 300]),
    http("Log healthy check", [1200, 420], "/actions", method="POST",
         body="={{ JSON.stringify({ action: 'drift_check_healthy', details: { drifted_features: $json.drifted_features, drifted_importance_mass: $json.drifted_importance_mass } }) }}"),
]
conns = connect(
    ("Every Monday 07:00", "Pull current customers"), ("Run manually", "Pull current customers"),
    ("Pull current customers", "Check drift"), ("Check drift", "Retrain recommended?"),
    ("Retrain recommended?", "Retrain model", 0), ("Retrain recommended?", "Log healthy check", 1),
    ("Retrain model", "Summarize retrain"),
    ("Summarize retrain", "Log retrain"), ("Summarize retrain", "Slack configured?"),
    ("Slack configured?", "Post to Slack"),
)
wf3 = workflow("Churn - weekly drift check & retrain", "ChurnDriftChk03", nodes, conns, [])

for fname, wf in [("01_retention_scoring.json", wf1), ("02_complaint_routing.json", wf2), ("03_drift_retrain.json", wf3)]:
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(wf, f, indent=2)
    print(fname, len(wf["nodes"]), "nodes")
