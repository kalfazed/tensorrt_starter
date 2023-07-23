-- Additional Plugins
lvim.plugins = {
  -- lsp related
  "ray-x/lsp_signature.nvim",
  -- DAP related
  "rcarriga/nvim-dap-ui",
  "mfussenegger/nvim-dap",
  "ldelossa/nvim-dap-projects",
  {
    "folke/trouble.nvim",
    cmd = "TroubleToggle",
  },
  -- markdown related
  {
    "iamcco/markdown-preview.nvim",
    build = "cd app && npm install",
    ft = "markdown",
    config = function()
      -- vim.g.mkdp_auto_start = 1
      vim.cmd [[
        let g:mkdp_open_to_the_world = 1
        " let g:mkdp_open_ip = '192.168.11.2'
        let g:mkdp_port = 3999
        let g:mkdp_browser = 'firefox'
        function! g:EchoUrl(url)
            :echo a:url
        endfunction
        let g:mkdp_browserfunc = 'g:EchoUrl'
      ]]
    end,
  },
  -- outline related
  {
    "simrat39/symbols-outline.nvim",
    config = function()
      require('symbols-outline').setup()
    end
  },
  -- chatGPT
  {
    "jackMort/ChatGPT.nvim",
    config = function()
      require("chatgpt").setup()
    end,
    requires = {
      "MunifTanjim/nui.nvim",
      "nvim-lua/plenary.nvim",
      "nvim-telescope/telescope.nvim"
    }
  }
}
