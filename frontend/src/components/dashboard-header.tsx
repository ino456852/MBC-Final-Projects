import { Button } from "@/components/ui/button"
import { MessageSquare, Menu, Moon, Sun } from "lucide-react"
import { Link } from "react-router-dom"

interface DashboardHeaderProps {
  onToggleLeftSidebar: () => void
  onToggleChatSidebar: () => void
}

export function DashboardHeader({ onToggleLeftSidebar, onToggleChatSidebar }: DashboardHeaderProps) {

  const toggleTheme = () => {
    const html = document.documentElement
    html.classList.toggle("dark")
  }

  return (
    <header className="border-b border-border bg-card">
      <div className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={onToggleLeftSidebar} className="p-2">
            <Menu className="w-4 h-4" />
          </Button>
          <h1 className="text-xl font-semibold text-foreground">미래환율팀</h1>
        </div>
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={onToggleChatSidebar} className="p-2">
            <MessageSquare className="w-4 h-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={toggleTheme} className="p-2">
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">테마 전환</span>
          </Button>
          <Link to="/login">
            <Button
              variant="outline"
              className="border-2 border-foreground text-foreground hover:bg-accent bg-transparent"
            >
              로그인
            </Button>
          </Link>
        </div>
      </div>
    </header>
  )
}
