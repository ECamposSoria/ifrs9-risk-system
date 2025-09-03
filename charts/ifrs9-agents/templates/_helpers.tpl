{{- define "ifrs9-agents.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ifrs9-agents.image" -}}
{{- printf "%s/%s:%s" .Values.image.registry .Chart.Name .Values.image.tag -}}
{{- end -}}

