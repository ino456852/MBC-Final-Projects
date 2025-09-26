import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send } from "lucide-react"

interface ChatSidebarProps {
  isOpen: boolean
}

export function ChatSidebar({ isOpen }: ChatSidebarProps) {
  return (
    <aside
      className={`bg-card border-l border-border transition-all duration-300 flex flex-col flex-shrink-0 h-full ${isOpen ? "w-80" : "w-0 overflow-hidden"
        }`}
    >
      <div className="w-80 h-full flex flex-col">
        <div className="p-4 border-b border-border flex-shrink-0">
          <h3 className="font-medium text-foreground">채팅기록</h3>
        </div>

        <div className="flex-1 p-4 overflow-hidden">
          <ScrollArea className="h-full pr-4">
            <div className="space-y-3">
              <div className="p-3 bg-accent rounded-lg">
                <p className="text-xs font-medium text-accent-foreground/80 mb-1">상담원</p>
                <p className="text-sm text-accent-foreground">안녕하세요! 환율 정보가 필요하신가요?</p>
                <span className="text-xs text-accent-foreground/70">오전 10:30</span>
              </div>
              <div className="p-3 bg-primary text-primary-foreground rounded-lg ml-8">
                <p className="text-xs font-medium opacity-80 mb-1">김철수</p>
                <p className="text-sm">네, 오늘 달러 환율을 확인하고 싶습니다.</p>
                <span className="text-xs opacity-70">오전 10:32</span>
              </div>
              <div className="p-3 bg-accent rounded-lg">
                <p className="text-xs font-medium text-accent-foreground/80 mb-1">상담원</p>
                <p className="text-sm text-accent-foreground">현재 USD/KRW 환율은 1,340원입니다.</p>
                <span className="text-xs text-accent-foreground/70">오전 10:33</span>
              </div>
            </div>
          </ScrollArea>
        </div>

        <div className="p-4 border-t border-border space-y-3 flex-shrink-0">
          <Textarea
            placeholder="메시지를 입력하세요..."
            className="min-h-[80px] resize-none"
          />
          <Button className="w-full" size="sm">
            <Send className="w-4 h-4 mr-2" />
            전송
          </Button>
        </div>
      </div>
    </aside>
  )
}
