# homeduplex Helm chart

Runs [homeduplex](https://github.com/thamdub/homeduplex) on Kubernetes: one Deployment, a Service, the settings in
a ConfigMap and, optionally, an Ingress.

```sh
helm install homeduplex oci://ghcr.io/thamdub/charts/homeduplex --version <version> -f my-values.yaml
```

## Values

The essential one is `config`: homeduplex's settings, exactly as in a `homeduplex.yaml` file
([`examples/homeduplex.example.yaml`](../../examples/homeduplex.example.yaml) lists them all). `stt`, `tts` and `llm`
are required; the chart refuses to render without them. The chart sets `server.host` and `server.port` itself.

```yaml
config:
  stt: {type: wyoming, uri: "tcp://stt.example.lan:10300"}
  tts: {type: wyoming, uri: "tcp://tts.example.lan:10200", voice: en_US-lessac-medium}
  llm: {type: ollama, url: "http://ollama.example.lan:11434", model: llama3.2:3b, options: {num_ctx: 8192}}
  prompt:
    context: "Today is {now:%A, %B %d, %Y}."
    timezone: Europe/Paris   # containers run in UTC; set this for {now}
```

| Value | Default | |
|---|---|---|
| `config` | `{}` | homeduplex settings (required: `stt`, `tts`, `llm`) |
| `existingSecret` | `""` | Secret whose keys become environment variables: `HOMEDUPLEX_LLM__API_KEY`, `HOMEDUPLEX_SERVER__API_KEY`, ... Keep secrets here, not in `config` |
| `extraEnv` | `[]` | more environment variables (for time zones, use `config.prompt.timezone`, not `TZ`) |
| `logLevel` | `INFO` | `DEBUG` also logs conversation text |
| `image.repository` / `image.tag` | `ghcr.io/thamdub/homeduplex` / the chart's appVersion | |
| `replicaCount` | `1` | backend `slots` are per process: more replicas multiply the load on each backend |
| `containerPort` / `service.port` | `8770` | |
| `service.type` / `service.annotations` | `ClusterIP` / `{}` | for example `LoadBalancer` with your load balancer's address annotation |
| `ingress.*` | disabled | a standard Ingress. Realtime clients use WebSockets; most controllers pass them through as is |
| `resources` | 50m CPU, 96Mi requested; 256Mi limit | measured use is about 40 MB and a few percent of a core |
| `podSecurityContext` / `securityContext` | non-root (UID 10001), read-only root filesystem, no capabilities | |
| `nodeSelector`, `tolerations`, `affinity`, `podAnnotations`, `podLabels`, `imagePullSecrets` | empty | |

Clients connect to `ws://<service or ingress address>/v1/realtime` (`wss://` behind TLS). homeduplex must be able to
reach your speech and model servers from inside the cluster, and your clients must be able to reach homeduplex.
