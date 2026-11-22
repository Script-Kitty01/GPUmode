package frontend

import (
	"embed"
	"net/http"
	"strings"
)

// Assets embeds the modular frontend CSS, HTML components, and JavaScript modules.
//
//go:embed css/*.css components/*.html js/*.js
var Assets embed.FS

var componentOrder = []string{
	"components/header.html",
	"components/telemetry_strip.html",
	"components/ebpf_boundary.html",
	"components/cuda_optimizer.html",
	"components/model_runner.html",
	"components/fleet_scheduler.html",
	"components/code_panels.html",
}

var scriptOrder = []string{
	"/static/js/utils.js",
	"/static/js/telemetry.js",
	"/static/js/optimizer.js",
	"/static/js/workload.js",
	"/static/js/scheduler.js",
	"/static/js/app.js",
}

// StaticHandler serves /static/css/* and /static/js/* directly from the modular frontend folder.
func StaticHandler() http.Handler {
	return http.StripPrefix("/static/", http.FileServer(http.FS(Assets)))
}

// RenderDashboardHTML assembles the modular HTML components from frontend/components/*.html
// and links the modular stylesheet (/static/css/dashboard.css) and JS modules (/static/js/*.js).
func RenderDashboardHTML() string {
	var b strings.Builder
	b.WriteString(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>gpuMode — Intelligent GPU Infrastructure & CUDA-to-Economics Platform</title>
<link rel="stylesheet" href="/static/css/dashboard.css">
</head>
<body>
`)
	for _, path := range componentOrder {
		if data, err := Assets.ReadFile(path); err == nil {
			b.Write(data)
			b.WriteByte('\n')
		}
	}
	for _, src := range scriptOrder {
		b.WriteString(`<script src="`)
		b.WriteString(src)
		b.WriteString(`"></script>`)
		b.WriteByte('\n')
	}
	b.WriteString("</body>\n</html>\n")
	return b.String()
}
