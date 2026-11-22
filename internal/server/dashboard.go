package server

import (
	"net/http"

	"gpumode/frontend"
)

func renderDashboardHTML() string {
	return frontend.RenderDashboardHTML()
}

func dashboardStaticHandler() http.Handler {
	return frontend.StaticHandler()
}
