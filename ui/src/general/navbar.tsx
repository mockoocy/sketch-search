import { RequireRole } from "@/auth/RequireRole";
import { Link } from "@tanstack/react-router";

export function Navbar() {
  return (
    <header className="border-b position-fixed top-0 z-50 w-vw bg-background/95 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4">
        <div className="font-semibold tracking-tight">
          <span className="text-2xl font-sketch text-foreground">sketch</span>
          <span className="mx-1">-</span>
          <span className="text-xl font-sans text-primary">search</span>
        </div>

        <nav className="flex items-center justify-between gap-6 text-sm">
          <RequireRole role="user">
            <Link
              to="/"
              className="text-muted-foreground hover:text-foreground"
              activeProps={{ className: "text-foreground font-medium" }}
            >
              Gallery
            </Link>
          </RequireRole>

          <RequireRole role="admin">
            <Link
              to="/admin"
              className="text-muted-foreground hover:text-foreground"
              activeProps={{ className: "text-foreground font-medium" }}
            >
              Admin
            </Link>
          </RequireRole>
        </nav>
      </div>
    </header>
  );
}
