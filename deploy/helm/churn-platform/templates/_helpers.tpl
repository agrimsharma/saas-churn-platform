{{- define "churn.name" -}}
{{- .Release.Name | trunc 40 | trimSuffix "-" -}}
{{- end -}}

{{- define "churn.labels" -}}
app.kubernetes.io/part-of: churn-platform
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "churn.selector" -}}
app.kubernetes.io/name: {{ .component }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
{{- end -}}

{{- define "churn.host" -}}
{{ index .root.Values.ingress.hosts .key }}.{{ .root.Values.ingress.domain }}
{{- end -}}

{{- define "churn.scheme" -}}
{{ ternary "https" "http" .Values.ingress.tls.enabled }}
{{- end -}}
