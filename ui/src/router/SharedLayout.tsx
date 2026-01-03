import { Toaster } from "@/general/components/sonner";
import { Navbar } from "@/general/navbar";
import { Outlet } from "@tanstack/react-router";

export function SharedLayout() {
  return (
    <div className="w-screen h-screen bg-gray-50 dark:bg-gray-900">
      <Navbar />
      <Outlet />
      <Toaster />
    </div>
  );
}
