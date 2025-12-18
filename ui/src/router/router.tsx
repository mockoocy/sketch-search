import { authRoute } from '@/auth/routes'
import { rootRoute } from '@/router/rootRoute'
import { createRouter } from '@tanstack/react-router'

const routeTree = rootRoute.addChildren([
  authRoute,
])

export const router = createRouter({
  routeTree,
})
