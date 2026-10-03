{{- define "homeduplex.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "homeduplex.fullname" -}}
{{- if contains .Chart.Name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "homeduplex.selectorLabels" -}}
app.kubernetes.io/name: {{ include "homeduplex.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "homeduplex.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "homeduplex.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- /* The settings file: the user's config, with the listening address owned by the chart. */ -}}
{{- define "homeduplex.config" -}}
{{- $config := deepCopy (.Values.config | default dict) -}}
{{- range $section := list "stt" "tts" "llm" -}}
{{- if not (hasKey $config $section) -}}
{{- fail (printf "config.%s is required (see examples/homeduplex.example.yaml)" $section) -}}
{{- end -}}
{{- end -}}
{{- $server := get $config "server" | default dict -}}
{{- $_ := set $server "host" "0.0.0.0" -}}
{{- $_ := set $server "port" (int .Values.containerPort) -}}
{{- $_ := set $config "server" $server -}}
{{- toYaml $config -}}
{{- end -}}
